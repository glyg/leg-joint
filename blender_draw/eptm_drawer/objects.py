import numpy as np
import bpy
import pandas as pd

cell_color = '#2887c8'
edge_color = '#3dc828'
j_radius = 0.3


def hex_to_rgb(col, factors = 255.):
    return tuple(c / factors for c in bytes.fromhex(col.replace('#', '')))

def set_jv(name, stamp, x, y, z):
    """

    """
    obj = bpy.data.objects.get(name)
    if obj  is None:
        bpy.ops.object.empty_add()
        obj = bpy.context.object
        obj.name = name
        obj.layers = tuple(i == 2 for i in range(20))
    scn = bpy.context.scene
    scn.frame_current = stamp
    obj.location = (x, y, z)
    obj.keyframe_insert('location')


def set_cell(name, x, y, z, stamp,
             j_xx, j_yy, j_zz, color):
    """
    """

    obj = bpy.data.objects.get(name)
    if obj  is None:
        me = bpy.data.meshes.new("CellMesh")
        obj = bpy.data.objects.new(name, me)
        bpy.context.scene.objects.link(obj)
        obj.layers = tuple(i == 1 for i in range(20))

    scn = bpy.context.scene
    scn.frame_current = stamp

    obj.location = (x, y, z)
    vertices = ([(0, 0, 0),]
                + [(jx, jy, jz) for jx, jy, jz in
                   zip(j_xx, j_yy, j_zz)]
                + [(j_xx[0], j_yy[0], j_zz[0])])
    faces = [(0, i + 1, i + 2, 0) for i in range(len(vertices) - 2)]
    me.from_pydata(vertices, [], faces)
    me.update(calc_edges=True)
    return obj

def create_junction_material():

    material = bpy.data.materials.new("junction")
    material.diffuse_color = hex_to_rgb(edge_color)


def set_junction_arm(name, src_idx, trgt_idx):
    '''
    '''
    if name in bpy.data.objects:
        return
    trgt_name = 'jv{}'.format(trgt_idx)
    src_name = 'jv{}'.format(src_idx)
    if not src_name in bpy.data.objects:
        return
    if not trgt_name in bpy.data.objects:
        return

    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=j_radius,
                                        view_align=False,
                                        enter_editmode=False,
                                        depth=1,
                                        location=(0, 0, 0))

    cyl_name = 'cyl{}to{}'.format(src_idx, trgt_idx)
    cyl = bpy.context.object
    cyl.name = cyl_name
    bpy.ops.object.material_slot_add()
    slot = cyl.material_slots[0]
    slot.material = bpy.data.materials["junction"]

    bpy.ops.object.armature_add()
    arm = bpy.context.object
    arm.location = (0, 0, 0)
    arm.name = name
    arm.layers = tuple(i == 3 for i in range(20))
    arm.parent = bpy.data.objects[src_name]
    arm.data.draw_type = 'STICK'
    bone = arm.pose.bones[0]
    bpy.ops.object.posemode_toggle()

    bpy.ops.pose.constraint_add(type='STRETCH_TO')
    bone.constraints["Stretch To"].target = bpy.data.objects[trgt_name]
    bone.constraints["Stretch To"].rest_length = 1.
    bone.constraints["Stretch To"].keep_axis = 'PLANE_Z'
    bone.constraints["Stretch To"].volume = 'NO_VOLUME'

    cyl.parent = bpy.data.objects[name]
    cyl.parent_type = 'BONE'
    cyl.parent_bone = "Bone"
    cyl.delta_location = (0, -0.5, 0)
    cyl.delta_rotation_euler = (np.pi/2, 0, 0)
    cyl.layers = tuple(i == 0 for i in range(20))


def set_junction(name, stamp, xx, yy, zz, color):
    """
    deprecated
    """
    scn = bpy.context.scene
    scn.frame_current = stamp
    obj = bpy.data.objects.get(name)
    if obj  is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.1,
                                            view_align=False,
                                            enter_editmode=False,
                                            depth=1,
                                            location=(0, 0, 0))
        obj = bpy.context.object
        obj.name = name
        obj.rotation_mode = 'AXIS_ANGLE'
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN',
                                  center='MEDIAN')

    x0, x1 = xx
    y0, y1 = yy
    z0, z1 = zz
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    depth = np.sqrt(dx**2 + dy**2 + dz**2)
    obj.scale = (1, 1, depth)
    obj.keyframe_insert('scale')
    axis_angle = (np.arctan2(np.sqrt(dx**2 + dy**2), dz), -dy, dx, 0)
    obj.rotation_axis_angle = axis_angle
    obj.keyframe_insert('rotation_axis_angle')
    location = ((x0 + x1)/2.,
                (y0 + y1)/2.,
                (z0 + z1)/2.)
    obj.location = location
    obj.keyframe_insert('location')
    material = bpy.data.materials.new("%s_color" % name)
    material.diffuse_color = hex_to_rgb(color)
    bpy.ops.object.material_slot_add()
    slot = obj.material_slots[0]
    slot.material = material

    return obj
