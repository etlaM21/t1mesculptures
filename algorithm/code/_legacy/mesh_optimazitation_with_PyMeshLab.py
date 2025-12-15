import pymeshlab
ms = pymeshlab.MeshSet()

ms.load_new_mesh('D:/wip/art/t1mesculptures/_gitrepo/vertReduce.stl')

for i in range(5):
    # Get the number of faces
    num_faces = ms.current_mesh().face_number()

    # Calculate the target number of faces after halving
    target_faces = num_faces // 10  # You can adjust this value as needed

    # Apply quadric edge collapse decimation using pymeshlab
    ms.apply_filter("meshing_decimation_quadric_edge_collapse", 
                    targetfacenum=target_faces,
                    targetperc=0.1, 
                    qualitythr=0.33,
                    boundaryweight=True, 
                    preservenormal=True, 
                    preservetopology=True, 
                    optimalplacement=False, 
                    planarquadric=True, 
                    planarweight=0.001, 
                    qualityweight=True, 
                    autoclean=True
                    )
    # ms.meshing_decimation_edge_collapse_for_marching_cube_meshes()
    ms.save_current_mesh('vertReduce_optimized_' + str(i) + ".stl")

