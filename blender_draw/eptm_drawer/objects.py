import numpy as np
import bpy
import pandas as pd

def hex_to_rgb(col, factors = 255.):
    return tuple(c / factors for c in bytes.fromhex(col.replace('#', '')))



def set_cell(name, x, y, z,
             j_xx, j_yy, j_zz, color):
    """
    """

    vertices = ([(0, 0, 0),]
                + [(jx, jy, jz) for jx, jy, jz in
                   zip(j_xx, j_yy, j_zz)]
                + [(j_xx[0], j_yy[0], j_zz[0])])
    faces = [(0, i + 1, i + 2, 0) for i in range(len(vertices) - 2)]
    me = bpy.data.meshes.new("CellMesh")
    obj = bpy.data.objects.new(name, me)
    obj.location = (x, y, z)
    me.from_pydata(vertices, [], faces)
    me.update(calc_edges=True)

    # size = (1., 1., 1.)

    # bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,
    #                                       size=1, view_align=False,
    # enter_editmode=False, location=(x, y, z), rotation=(0, 0, 0))
    bpy.context.scene.objects.link(obj)
    obj.layers = tuple(i == 1 for i in range(20))

    # material = bpy.data.materials.new("%s_color" % name)
    # material.diffuse_color = hex_to_rgb(color)
    # bpy.ops.object.material_slot_add()
    # slot = obj.material_slots[0]
    # slot.material = material
    return obj

def set_junction(name, xx, yy, zz, color):
    """
    """

    x0, x1 = xx
    y0, y1 = yy
    z0, z1 = zz
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    depth = np.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.1,
                                        view_align=False,
                                        enter_editmode=False,
                                        depth=depth,
                                        location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    # obj.dimensions = (1, 1, 1)

    # obj.rotation_mode = 'ZYX'
    # alpha = np.arctan2(dx, dy)# -
    # beta = 0#np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    # gamma = 0#np.pi -  np.arctan2(dx, dy)
    # rotation = (alpha, beta, gamma)
    # obj.rotation_euler = rotation

    # obj.rotation_mode = 'QUATERNION'
    # quaternion = (dz, dy, -dx, 0)
    # obj.rotation_quaternion = quaternion

    obj.rotation_mode = 'AXIS_ANGLE'
    axis_angle = (np.arctan2(np.sqrt(dx**2 + dy**2), dz), -dy, dx, 0)
    obj.rotation_axis_angle = axis_angle


    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN',
                              center='MEDIAN')

    location = ((x0 + x1)/2.,
                (y0 + y1)/2.,
                (z0 + z1)/2.)
    obj.location = location
    material = bpy.data.materials.new("%s_color" % name)
    material.diffuse_color = hex_to_rgb(color)
    bpy.ops.object.material_slot_add()
    slot = obj.material_slots[0]
    slot.material = material

    return obj
