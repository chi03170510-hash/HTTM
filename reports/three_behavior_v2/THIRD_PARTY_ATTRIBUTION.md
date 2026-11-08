# Third-party data attribution for HTTM ThreeBehavior V2

The Kaggle V2 training package was built from these public Roboflow dataset versions, whose source `data.yaml` files declare CC BY 4.0:

- Behaviour Detection, *Classroom Behavior*, version 1: https://universe.roboflow.com/behaviour-detection-tlowy/classroom-behavior-05jvk/dataset/1
- Classroom, *Classroom Student Dataset*, version 16: https://universe.roboflow.com/classroom-na2vo/classroom-student-dataset/dataset/16

The trained V2 checkpoint is a modified model trained from the source annotations with the mapping `sleep`/`Sleeping` → `sleeping`, `Using Phone` → `using_phone`, and `Hand Rising` → `raising_hand`. The ambiguous `hand` class and unrelated classes were not mapped. The dataset packages themselves are not redistributed in this GitHub repository.

The sources declare Creative Commons Attribution 4.0. Redistribution under that license requires appropriate credit, a link to the license, and an indication of changes; see https://creativecommons.org/licenses/by/4.0/ . No source author endorses HTTM by this attribution. The checkpoint release includes this notice. This records the source metadata and attribution applied; it is not a legal opinion about rights that dataset uploaders may not control, image/person rights, or all downstream uses.
