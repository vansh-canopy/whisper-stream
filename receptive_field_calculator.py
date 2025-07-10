import math

def calculate_receptive_field_step(layer_type, kernel, pad, stride, dilation, J_prev, L_prev, beta_prev):    
    if layer_type == "Tconv":
        J_curr = J_prev / stride    
        L_curr = L_prev - J_curr * math.floor((kernel - 1)) / stride
        beta_curr = beta_prev + pad * J_curr
    else:
        J_curr = J_prev * stride
        beta_curr = beta_prev
        L_curr = L_prev - (kernel - 1) * dilation * J_prev

    return J_curr, L_curr, beta_curr


def calculate_sequence(layers_config, initial_J=1.0, initial_L=1.0, initial_beta=0.0):  
    J = [initial_J]
    L = [initial_L] 
    beta = [initial_beta]
    
    for i, config in enumerate(layers_config):
        try:
            J_next, L_next, beta_next = calculate_receptive_field_step(
                config['type'],
                config['kernel'],
                config['pad'],
                config['stride'], 
                config.get('dilation', 1),
                J[-1], L[-1], beta[-1]
            )
            
            J.append(J_next)
            L.append(L_next)
            beta.append(beta_next)
            
        except ValueError as e:
            print(f"Error at layer {i}: {e}")
            break
            
    return J, L, beta


if __name__ == "__main__":
    dilations = [1, 3, 9]
    paddings = [(6 *dilation // 2) for dilation in dilations]
    
    layers = [
        {'type': 'Tconv', 'kernel': 16, 'pad': 4, 'stride': 8, 'dilation': 1},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[0], 'stride': 1, 'dilation': dilations[0]},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[1], 'stride': 1, 'dilation': dilations[1]},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[2], 'stride': 1, 'dilation': dilations[2]},
        
        {'type': 'Tconv', 'kernel': 16, 'pad': 4, 'stride': 8, 'dilation': 1},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[0], 'stride': 1, 'dilation': dilations[0]},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[1], 'stride': 1, 'dilation': dilations[1]},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[2], 'stride': 1, 'dilation': dilations[2]},
        
        {'type': 'Tconv', 'kernel': 8, 'pad': 2, 'stride': 4, 'dilation': 1},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[0], 'stride': 1, 'dilation': dilations[0]},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[1], 'stride': 1, 'dilation': dilations[1]},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[2], 'stride': 1, 'dilation': dilations[2]},
        
        {'type': 'Tconv', 'kernel': 4, 'pad': 1, 'stride': 2, 'dilation': 1},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[0], 'stride': 1, 'dilation': dilations[0]},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[1], 'stride': 1, 'dilation': dilations[1]},
        {'type': 'conv', 'kernel': 7, 'pad': paddings[2], 'stride': 1, 'dilation': dilations[2]},
    ]
    
    J, L, beta = calculate_sequence(layers, initial_J=1.0, initial_L=224.0, initial_beta=0.0)
    
    for i, (j, l, b) in enumerate(zip(J, L, beta)):
        print(f"Layer {i}: J={j:.7f}, L={l:.7f}, beta={b:.7f}") 