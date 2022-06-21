import torch
import torch.distributed as dist
from collections import defaultdict
import time

class HookManager:
    def __init__(self):
        self.hooks = defaultdict(list)
        self.step_cron_gcd = 100000000000000

    def clear_all_hooks(self):
        self.hooks = defaultdict(list)

    def clear_hooks(self, hook_type):
        self.hooks[hook_type] = []

    def add_hook(self, hook_type, func):
        self.hooks[hook_type].append((func, None))

    def _run_hooks(self, hook_type):
        for hook, _ in self.hooks[hook_type]:
            hook()

    def add_pre_epoch_hook(self, func):
        self.add_hook('pre_epoch', func)

    def add_post_epoch_hook(self, func):
        self.add_hook('post_epoch', func)

    def _run_pre_epoch_hooks(self):
        self._run_hooks('pre_epoch')

    def _run_post_epoch_hooks(self):
        self._run_hooks('post_epoch')

    def add_cron_hooks(self, func, num_seconds, distributed=False):
        """
        Schedule a function to be run every num_seconds
        """
        self.hooks['cron'].append([func, [num_seconds, time.time(), distributed]])

    def _run_cron_hooks(self):
        # Info contains how often to run and last run time
        for hook, info in self.hooks['cron']:
            if info[2] is False:
                if (time.time() - info[1]) >= info[0]:
                    hook()
            else:
                t = torch.tensor(1. if (time.time() - info[1]) >= info[0] else 0.).cuda()
                dist.all_reduce(t)
                if t.cpu().item() > 0:
                    hook()

    def get_step_cron_gcd(self):
        """
        Computes how often the step_cron should be run by finding the gcd of all entries.

        Returns - gcd of how frequently every step_cron hook wants to be run.
        """

        if len(self.hooks['step_cron']) == 0:
            return 1

        def gcd(a,b):
            while b>0:
                a, b = b, a%b
            return a

        result = self.hooks['step_cron'][0][1]
        for h in self.hooks['step_cron'][1:]:
            result = gcd(result, h[1])
        self.step_cron_gcd = result
        return result

    def add_step_cron_hooks(self, func, num_steps, reject_ok=False):
        """
        Registers func to run every num_steps

        Arguments:
        - func - a function to be run with function signature (epoch, step, batch, net_out)
        - num_steps - number of absolute steps taken from the start
        - reject_ok - If True soft reject. If False raise an error
        """
        if num_steps <= 0 and reject_ok:
            return
        elif num_steps <= 0:
            raise RuntimeError("Invalid request to a step_cron - %s @ %d" % (str(fund), num_steps))

        self.hooks['step_cron'].append((func, num_steps))
        self.get_step_cron_gcd()

    def _run_step_cron_hooks(self, cur_step):
        for hook, num_steps in self.hooks['step_cron']:
            if cur_step % num_steps == 0:
                hook()

    def add_start_hook(self, func):
        """
        Add a function to be run before the start of training
        """
        self.add_hook('start', func)

    def _run_start_hooks(self):
        self._run_hooks('start')

    def add_finalize_hook(self, func):
        """
        Add a function to be run after training is done
        """
        self.add_hook('finalize', func)

    def _run_finalize_hooks(self):
        self._run_hooks('finalize')
