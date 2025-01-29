import torch as th

from timelens.common import pytorch_tools


def compute_source_coordinates(y_displacement, x_displacement):
    """Retruns source coordinates, given displacements.
    
    Given traget coordinates (y, x), the source coordinates are
    computed as (y + y_displacement, x + x_displacement).
    
    Args:
        x_displacement, y_displacement: are tensors with indices 
                                        [example_index, 1, y, x]
    """
    device = x_displacement.device  # Usa il device di x_displacement

    width, height = y_displacement.size(-1), y_displacement.size(-2)
    x_target, y_target = pytorch_tools.create_meshgrid(width, height)

    # Sposta x_target e y_target sullo stesso device di x_displacement
    x_target = x_target.to(device)
    y_target = y_target.to(device)

    # Ora tutti i tensori sono sullo stesso device
    x_source = x_target + x_displacement.squeeze(1)
    y_source = y_target + y_displacement.squeeze(1)

    # Anche la mask deve essere su MPS
    out_of_boundary_mask = ((x_source.detach() < 0) | (x_source.detach() >= width) |
                            (y_source.detach() < 0) | (y_source.detach() >= height))

    return y_source, x_source, out_of_boundary_mask.to(device)


def backwarp_2d(source, y_displacement, x_displacement):
    """Returns warped source image and occlusion_mask.
    Value in location (x, y) in output image in taken from
    (x + x_displacement, y + y_displacement) location of the source image.
    If the location in the source image is outside of its borders,
    the location in the target image is filled with zeros and the
    location is added to the "occlusion_mask".

    Args:
        source: is a tensor with indices
                [example_index, channel_index, y, x].
        x_displacement,
        y_displacement: are tensors with indices [example_index,
                        1, y, x].
    Returns:
        target: is a tensor with indices
                [example_index, channel_index, y, x].
        occlusion_mask: is a tensor with indices [example_index, 1, y, x].
    """
    width, height = source.size(-1), source.size(-2)
    y_source, x_source, out_of_boundary_mask = compute_source_coordinates(y_displacement, x_displacement)

    # Assicura che tutte le variabili siano sullo stesso device
    device = source.device  # Prendi il device del source
    y_source = y_source.to(device)
    x_source = x_source.to(device)
    out_of_boundary_mask = out_of_boundary_mask.to(device)

    # Normalizza le coordinate tra -1 e 1
    x_source = (2.0 / float(width - 1)) * x_source - 1
    y_source = (2.0 / float(height - 1)) * y_source - 1

    x_source = x_source.masked_fill(out_of_boundary_mask, 0)
    y_source = y_source.masked_fill(out_of_boundary_mask, 0)

    grid_source = th.stack([x_source, y_source], -1).to(device)

    # Sposta source sullo stesso device di grid_source
    source = source.to(grid_source.device)

    target = th.nn.functional.grid_sample(source, grid_source, align_corners=True)

    out_of_boundary_mask = out_of_boundary_mask.unsqueeze(1)
    target.masked_fill_(out_of_boundary_mask.expand_as(target), 0)

    return target, out_of_boundary_mask
