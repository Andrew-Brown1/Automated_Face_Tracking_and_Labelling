"""
this is the pipeline to do everything - go from a directory of images to "cleaned" feature vectors 

(1) detect the faces - in Full_tracker_pipeline/src

(2) extract features - /scratch/shared/beegfs/abrown/BBC_work/ICMR_BL/Extract_Features

(3) do the cleaning + decide famous / non-famous - in the /scratch/shared/beegfs/abrown/BBC_work/ICMR_BL directory 

(4) save the dictionary 

(5) re-use everything possible from the video pipeline, and maybe objectify some stuff

(6) track the videos 

(6.5) get the annotations into some format for the annotating

(7) do the annotation - with the query expansion as well (ignore the voice and non-famous people for now)

(8) visualise the annotations

(9) think a lot about the output format 
"""


