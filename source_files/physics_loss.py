# Fluid quantities
nu = 1
rho = 1050

# def physics_loss(input_tensor, output_tensor):
#
#     grads = torch.autograd.grad(output_tensor, input_tensor, grad_outputs=torch.ones_like(output_tensor), create_graph=True)
#     print("help")
    # x = input_tensor[:, 0]
    # y = input_tensor[:, 1]
    # z = input_tensor[:, 2]
    #
    # u = output_tensor[:, 0]
    # v = output_tensor[:, 1]
    # w = output_tensor[:, 2]
    # p = output_tensor[:, 3]

    # u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    #
    # v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    # v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    # v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    #
    # w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    # w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    # w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    #
    # incompressibility_error = u_x + v_y + w_z
    #
    # p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    # p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    # p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    #
    # u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    # u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    # u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    #
    # v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    # v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    # v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
    #
    # w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    # w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    # w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), create_graph=True)[0]
    #
    # momentum_conservation_error_x = u * u_x + v * u_y + w * u_z - (-1 / rho * p_x + nu * (u_xx + u_yy + u_zz))
    # momentum_conservation_error_y = u * v_x + v * v_y + w * v_z - (-1 / rho * p_y + nu * (v_xx + v_yy + v_zz))
    # momentum_conservation_error_z = u * u_x + v * u_y + w * u_z - (-1 / rho * p_z + nu * (w_xx + w_yy + w_zz))
    #
    # return incompressibility_error + momentum_conservation_error_x + momentum_conservation_error_y + momentum_conservation_error_z
