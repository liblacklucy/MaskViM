import torch

from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS
from mmengine.hooks import Hook


@HOOKS.register_module()
class ExpHook(Hook):
    priority = 'NORMAL'

    def before_run(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        # num_parameters = sum([p.numel() for _, p in model.named_parameters() if p.requires_grad is True])
        # print_log(f'#Params (learnable): {num_parameters}')


        for name, parms in model.named_parameters():
            print('-->name:', name)
            # if 'token_mask' in name:
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")

        num_parameters = sum([p.numel() for p in model.parameters()])
        print_log(f'#Params: {num_parameters}')  # tiny:60893702 | 60606982     small:82491398      base:88551078 or 115923366

    # def after_train_iter(self,
    #                      runner,
    #                      batch_idx: int,
    #                      data_batch=None,
    #                      outputs=None) -> None:
    #     model = runner.model
    #     if is_model_wrapper(model):
    #         model = model.module
    #
    #     # num_parameters = sum([p.numel() for _, p in model.named_parameters() if p.requires_grad is True])
    #     # print_log(f'#Params (learnable): {num_parameters}')
    #     # num_parameters = sum([p.numel() for p in model.parameters()])
    #     # print_log(f'#Params: {num_parameters}')
    #
    #     for name, parms in model.named_parameters():
    #         # print(name)
    #         if 'dt_projs_weight' in name:
    #             print(torch.equal(parms.grad, torch.zeros_like(parms.grad)))
    #             print('-->name:', name)
    #             # print('-->para:', parms)
    #             # print('-->grad_requirs:', parms.requires_grad)
    #             # print('-->grad_value:', parms.grad)
    #             print("===")
