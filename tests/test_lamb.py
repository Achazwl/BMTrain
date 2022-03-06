import apex
import torch
import bmtrain as bmt
import time

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 128)
        self.param = torch.nn.Parameter(torch.empty(1237))

from apex.multi_tensor_apply import multi_tensor_applier
class RefLAMB(torch.optim.Optimizer):
    "from apex/tests/L0/run_optimizers/test_lamb.py"
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01)
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RefLAMB, self).__init__(params, defaults)
        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_l2norm=amp_C.multi_tensor_l2norm
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=self.param_groups[0]["params"][0].device)
            self.multi_tensor_lamb = amp_C.multi_tensor_lamb
        else:
            raise RuntimeError('apex.optimizers.FusedLAMB requires cuda extensions')

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError('FusedLAMB only support fp16 and fp32.')

        device = self.param_groups[0]["params"][0].device
        g_norm_32, g_norm_16 = torch.zeros(1, device=device), torch.zeros(1, device=device)
        # compute grad norm for two lists
        if len(g_all_32) > 0:
            g_norm_32 = multi_tensor_applier(self.multi_tensor_l2norm,
                                             self._dummy_overflow_buf,
                                             [g_all_32], False)[0]
        if len(g_all_16) > 0:
            g_norm_16 = multi_tensor_applier(self.multi_tensor_l2norm,
                                             self._dummy_overflow_buf,
                                             [g_all_16], False)[0]

        clipped_ratio = 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data *= clipped_ratio
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)

                m_t, v_t = state['m'], state['v']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # m_t = beta1 * m + (1 - beta1) * g_t
                m_t.mul_(beta1).add_(grad, alpha=1-beta1)
                # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
                v_t.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # Debiasing
                m_t_hat = m_t / (1.0 - beta1 ** state['step'])
                v_t_hat = v_t / (1.0 - beta2 ** state['step'])

                update = m_t_hat / v_t_hat.sqrt().add(group['eps'])

                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                trust_ratio = 1.0
                w_norm = p.data.pow(2).sum().sqrt()
                g_norm = update.pow(2).sum().sqrt()
                if w_norm > 0 and g_norm > 0:
                    trust_ratio = w_norm / g_norm

                state['w_norm'] = w_norm
                state['g_norm'] = g_norm
                state['trust_ratio'] = trust_ratio

                step_size = group['lr']

                p.data.add_(update, alpha=-step_size*trust_ratio)

        return loss

def main():
    model1 = TestModule()
    model2 = TestModule()
    model3 = TestModule()

    state_dict = model1.state_dict()
    for kw in state_dict.keys():
        state_dict[kw] = torch.randn_like(state_dict[kw])
    
    model1.load_state_dict(state_dict)
    model2.load_state_dict(state_dict)
    model3.load_state_dict(state_dict)

    model1 = model1.cuda().half()
    model2 = model2.cuda().half()
    model3 = model3.cuda()
    
    opt1 = bmt.optim.LambOptimizer(model1.parameters(), weight_decay=1e-3, scale=1)
    opt2 = bmt.optim.LambOffloadOptimizer(model2.parameters(), weight_decay=1e-3, scale=1)
    opt3 = RefLAMB(model3.parameters(), weight_decay=1e-3)

    for _ in range(100):
        opt1.zero_grad()
        opt2.zero_grad()
        opt3.zero_grad()

        for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
            grad = torch.randn_like(p1)
            p1.grad = grad
            p2.grad = grad
            p3.grad = grad.float()
        
        opt1.step()
        opt2.step()
        opt3.step()

        for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
            diff1 = torch.abs(p1 - p2).max()
            diff2 = torch.abs(p1 - p3).max()
            diff3 = torch.abs(p2 - p3).max()
            print(diff1, diff2, diff3)

if __name__ == "__main__":
    main()
