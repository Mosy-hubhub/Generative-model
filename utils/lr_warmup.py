def get_current_lr_linear(step, warmup_steps = 5000, max_lr = 1e-4):
    """
    linear lr warmup
    
    step: global training step
    """
    if step < warmup_steps:
        return max_lr * ((step + 1) / warmup_steps)
    return max_lr