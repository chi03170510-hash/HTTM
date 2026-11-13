import sys
import unittest
import json
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path

AI=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(AI))
from src.hybrid_v7 import (_phone_owner, TemporalV6Fusion, decide_sleeping_candidate,
                           decide_phone_candidate_v6_1, assess_food_head_v6_1, phone_surface_candidate,
                           deduplicate_sleeping_people, phone_object_geometry_plausible)
from src.hybrid_v7 import (decide_phone_candidate, assess_food_head, apply_ambiguous_sleep_phone_conflicts_v7,
                           _mouth_and_head_v7, person_box_supports_highres_behavior_v7,
                           food_crop_candidate_score_eligible_v7)


class TestHybridV7(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((AI/"config"/"hybrid_v7.json").read_text(encoding="utf-8"))

    def test_phone_associates_without_hand_landmarks(self):
        people=[{"detection_id":"student-a","bbox_xyxy":[0,0,100,200]},
                {"detection_id":"student-b","bbox_xyxy":[150,0,250,200]}]
        phone={"bbox_xyxy":[65,140,80,160]}
        result=_phone_owner(phone,people,self.cfg)
        self.assertEqual(result["status"],"ASSOCIATED_GEOMETRICALLY")
        self.assertEqual(result["person_detection_id"],"student-a")

    def test_sleep_yolo_candidate_does_not_require_pose_face_or_visible_body(self):
        state,_=decide_sleeping_candidate(.81,[10,10,80,80],self.cfg)
        self.assertEqual(state,"detected")

    def test_adjacent_sleeping_boxes_remain_independent(self):
        a=[0,10,40,90];b=[42,10,82,90]
        da=decide_sleeping_candidate(.80,a,self.cfg)[0];db=decide_sleeping_candidate(.82,b,self.cfg)[0]
        self.assertEqual((da,db),("detected","detected"));self.assertNotEqual(a,b)

    def test_phone_yolo_plus_associated_phone_confirms_without_hand(self):
        self.assertEqual(decide_phone_candidate(.30,True,False,self.cfg)[0],"detected")

    def test_phone_without_object_is_only_suspected_at_candidate_score(self):
        self.assertEqual(decide_phone_candidate(.84,False,False,self.cfg)[0],"suspected")

    def test_phone_for_other_person_does_not_confirm(self):
        people=[{"detection_id":"other","bbox_xyxy":[0,0,100,100]}]
        assoc=_phone_owner({"bbox_xyxy":[30,30,50,50]},people,self.cfg)
        self.assertNotEqual(assoc["person_detection_id"],"target")
        self.assertEqual(decide_phone_candidate(.84,False,True,self.cfg)[0],"suspected")

    def test_phone_on_desk_without_yolo_candidate_does_not_confirm(self):
        self.assertEqual(decide_phone_candidate(None,True,False,self.cfg)[0],"unknown")

    def test_phone_below_candidate_threshold_does_not_confirm_from_object_alone(self):
        self.assertEqual(decide_phone_candidate(.20,True,False,self.cfg)[0],"unknown")

    def test_flat_low_phone_without_nearby_wrist_stays_suspected(self):
        phone={"bbox_xyxy":[313,916,518,1024]};person=[2,399,528,1278]
        self.assertTrue(phone_surface_candidate(phone,person,{},self.cfg))
        self.assertEqual(decide_phone_candidate(.88,True,False,self.cfg,surface_conflict=True)[0],"suspected")

    def test_low_phone_near_face_is_not_treated_as_resting_on_desk(self):
        phone={"bbox_xyxy":[402,275,456,311]};person=[257,146,519,399]
        pose={"nose":{"x":429,"y":250,"visibility":.99}}
        self.assertFalse(phone_surface_candidate(phone,person,pose,self.cfg))

    def test_phone_near_face_does_not_require_wrist_evidence(self):
        phone={"bbox_xyxy":[90,20,120,90]};person=[0,0,200,500]
        self.assertFalse(phone_surface_candidate(phone,person,{},self.cfg))
        self.assertEqual(decide_phone_candidate(.70,True,False,self.cfg)[0],"detected")

    def test_low_phone_candidate_near_sleeping_student_without_wrist_stays_suspected(self):
        self.assertEqual(decide_phone_candidate(.35,True,False,self.cfg,sleeping_conflict=True)[0],"suspected")

    def test_food_near_visible_mouth_is_eating_evidence(self):
        foods=[{"detection_id":"food1","class_name":"sandwich","confidence":.3,"bbox_xyxy":[40,40,60,60]}]
        state,_,matches,_=assess_food_head(foods,{"food1":"student"},"student",[50,50],100,True,self.cfg)
        self.assertEqual(state,"detected");self.assertEqual(len(matches),1)

    def test_food_on_desk_is_not_eating(self):
        foods=[{"detection_id":"food1","class_name":"apple","confidence":.9,"bbox_xyxy":[40,150,60,170]}]
        state,_,_,_=assess_food_head(foods,{"food1":"student"},"student",[50,30],100,True,self.cfg)
        self.assertEqual(state,"not_detected")

    def test_food_associated_to_another_person_does_not_count(self):
        foods=[{"detection_id":"food1","class_name":"banana","confidence":.9,"bbox_xyxy":[40,40,60,60]}]
        state,_,_,_=assess_food_head(foods,{"food1":"other"},"student",[50,50],100,True,self.cfg)
        self.assertEqual(state,"unknown")

    def test_bottle_and_cup_are_never_food_evidence(self):
        for cls in ("bottle","cup","fork","spoon"):
            state,_,_,_=assess_food_head([{"detection_id":"x","class_name":cls,"bbox_xyxy":[40,40,60,60]}],
                {"x":"student"},"student",[50,50],100,True,self.cfg)
            self.assertEqual(state,"unknown")

    def test_head_fallback_near_food_is_suspected_and_no_food_is_not_fabricated(self):
        food={"detection_id":"f","class_name":"cake","bbox_xyxy":[40,40,60,60]}
        state,_,_,_=assess_food_head([food],{"f":"student"},"student",[50,50],100,False,self.cfg)
        self.assertEqual(state,"suspected")
        state,_,_,_=assess_food_head([],{},"student",[50,50],100,False,self.cfg)
        self.assertEqual(state,"unknown")

    def test_phone_between_competing_people_stays_ambiguous(self):
        people=[{"detection_id":"a","bbox_xyxy":[0,0,100,200]},
                {"detection_id":"b","bbox_xyxy":[100,0,200,200]}]
        result=_phone_owner({"bbox_xyxy":[92,140,108,160]},people,self.cfg)
        self.assertNotEqual(result["status"],"ASSOCIATED_GEOMETRICALLY")

    def test_original_box_containment_beats_neighbor_expansion(self):
        people=[{"detection_id":"owner","bbox_xyxy":[0,0,100,200]},
                {"detection_id":"neighbor","bbox_xyxy":[110,0,210,200]}]
        result=_phone_owner({"bbox_xyxy":[85,80,95,100]},people,self.cfg)
        self.assertEqual(result["person_detection_id"],"owner")
        self.assertTrue(result["phone_center_inside_original_person_box"])

    def test_video_smoothing_requires_two_samples_and_is_track_scoped(self):
        fusion=TemporalV6Fusion(self.cfg)
        behavior={b:{"state":"unknown","model_confidence":None,"decision_reason":"none","evidence":{}} for b in ("sleeping","using_phone","turning_back","eating")}
        behavior["sleeping"]={"state":"detected","model_confidence":.8,"decision_reason":"yolo","evidence":{}}
        first=fusion.update(1,0.,behavior)
        second=fusion.update(1,1.,behavior)
        other=fusion.update(2,1.,{**behavior,"sleeping":{"state":"unknown","model_confidence":None,"decision_reason":"none","evidence":{}}})
        self.assertEqual(first["sleeping"]["state"],"suspected")
        self.assertEqual(second["sleeping"]["state"],"detected")
        self.assertEqual(other["sleeping"]["state"],"unknown")

    def test_same_person_yolo_and_phone_object_confirms_without_wrist(self):
        self.assertEqual(decide_phone_candidate_v6_1(.31,"ASSOCIATED_GEOMETRICALLY",True,self.cfg),
                         ("detected","YOLO_PHONE_AND_SAME_PERSON_COCO_OBJECT"))

    def test_phone_object_miss_keeps_yolo_candidate_suspected(self):
        self.assertEqual(decide_phone_candidate_v6_1(.82,None,False,self.cfg),
                         ("suspected","PHONE_OBJECT_NOT_DETECTED"))

    def test_phone_association_ambiguity_stays_suspected(self):
        self.assertEqual(decide_phone_candidate_v6_1(.74,"AMBIGUOUS",True,self.cfg),
                         ("suspected","PHONE_ASSOCIATION_AMBIGUOUS"))

    def test_phone_object_box_as_large_as_a_book_does_not_confirm(self):
        valid,_=phone_object_geometry_plausible([1032,368,1133,510],[995,190,1278,712],self.cfg)
        self.assertFalse(valid)

    def test_handheld_phone_scale_remains_plausible(self):
        valid,_=phone_object_geometry_plausible([1249,1013,1351,1058],[1005,528,1595,1120],self.cfg)
        self.assertTrue(valid)

    def test_phone_associated_to_other_person_does_not_confirm(self):
        self.assertEqual(decide_phone_candidate_v6_1(.85,"ASSOCIATED_TO_OTHER_PERSON",True,self.cfg),
                         ("suspected","PHONE_ASSOCIATION_FAILED"))

    def test_sleeping_duplicate_requires_person_and_behavior_box_overlap(self):
        def person(pid,pbox,sbox,score):
            return {"person_ref":pid,"person_bbox_xyxy_clipped":pbox,"behaviors":{"sleeping":{
                "state":"detected","status":"IMAGE_ONLY","decision":"detected","model_confidence":score,
                "bounding_box":sbox,"evidence":{"sleeping_yolo_box":sbox}}}}
        a=person("a",[0,0,100,200],[10,10,90,100],.8)
        b=person("b",[2,2,102,202],[11,11,91,101],.9)
        c=person("c",[104,0,204,200],[114,10,194,100],.88)
        removed=deduplicate_sleeping_people([a,b,c],self.cfg)
        self.assertEqual(len(removed),1)
        self.assertEqual(a["behaviors"]["sleeping"]["state"],"unknown")
        self.assertEqual(b["behaviors"]["sleeping"]["state"],"detected")
        self.assertEqual(c["behaviors"]["sleeping"]["state"],"detected")

    def test_food_near_visible_mouth_confirms_only_same_person_object(self):
        food={"detection_id":"f","class_name":"sandwich","confidence":.4,"bbox_xyxy":[45,45,60,60]}
        state,_,matches,_=assess_food_head_v6_1([food],{"f":"student"},"student",[50,50],60,True,self.cfg)
        self.assertEqual(state,"detected");self.assertEqual(matches[0]["food_detection_id"],"f")

    def test_pose_or_head_proxy_remains_suspected_and_desk_food_is_negative(self):
        food={"detection_id":"f","class_name":"cake","confidence":.4,"bbox_xyxy":[45,45,60,60]}
        state,_,_,_=assess_food_head_v6_1([food],{"f":"student"},"student",[50,50],60,False,self.cfg)
        self.assertEqual(state,"suspected")
        desk={"detection_id":"d","class_name":"apple","confidence":.8,"bbox_xyxy":[45,150,60,170]}
        state,_,_,_=assess_food_head_v6_1([desk],{"d":"student"},"student",[50,50],60,True,self.cfg)
        self.assertEqual(state,"not_detected")

    def test_food_not_associated_to_this_student_is_unknown(self):
        food={"detection_id":"f","class_name":"banana","confidence":.8,"bbox_xyxy":[45,45,60,60]}
        state,_,_,_=assess_food_head_v6_1([food],{"f":"other"},"student",[50,50],60,True,self.cfg)
        self.assertEqual(state,"unknown")

    def test_ambiguous_phone_object_demotes_sleep_only_and_does_not_confirm_phone(self):
        person={"person_ref":"student-a","person_bbox_xyxy_clipped":[10,10,100,200],"bbox_xyxy":[10,10,100,200],
            "behaviors":{"sleeping":{"state":"detected","status":"IMAGE_ONLY","decision":"detected",
                "bounding_box":[20,20,95,180],"contradicting_evidence":[],"evidence":{"sleeping_yolo_box":[20,20,95,180]}},
                "using_phone":{"state":"unknown","decision":"unknown"}}}
        out={"people":[person],"phone_detections":[{"detection_id":"ph","bbox_xyxy":[70,160,88,180],"confidence":.3,"source":"person_crop_coco"}],
            "v6_1_phone_associations":[{"phone_detection_id":"ph","person_detection_id":"neighbor","candidate_person_ids":["neighbor","student-a"],"status":"AMBIGUOUS"}]}
        conflicts=apply_ambiguous_sleep_phone_conflicts_v7(out,self.cfg)
        self.assertEqual(len(conflicts),1)
        self.assertEqual(person["behaviors"]["sleeping"]["state"],"suspected")
        self.assertEqual(person["behaviors"]["using_phone"]["state"],"unknown")
        self.assertTrue(person["behaviors"]["sleeping"]["evidence"]["phone_object_does_not_confirm_using_phone"])

    def test_opening_semantics_do_not_turn_food_or_ear_score_into_action_probability(self):
        person={"bbox_xyxy":[0,0,100,200],"evidence":{"face":{"mouth_center_xy_frame":[50,30]},"pose_landmarks":{}}}
        mouth,head_h,valid=_mouth_and_head_v7(person)
        self.assertEqual(mouth,[50.,30.]);self.assertGreaterEqual(head_h,24.);self.assertTrue(valid)

    def test_highres_sleep_box_needs_nonpartial_person_anchor(self):
        self.assertFalse(person_box_supports_highres_behavior_v7([0,0,58,43],self.cfg))
        self.assertFalse(person_box_supports_highres_behavior_v7([0,0,65,27],self.cfg))
        self.assertTrue(person_box_supports_highres_behavior_v7([0,0,70,90],self.cfg))

    def test_weak_food_crop_object_score_is_not_spatial_eating_evidence(self):
        self.assertFalse(food_crop_candidate_score_eligible_v7(.045,self.cfg))
        self.assertTrue(food_crop_candidate_score_eligible_v7(.10,self.cfg))


class TestV7PipelineRules(unittest.TestCase):
    """Exercise final V7 decisions, not only the standalone rule helpers."""

    def setUp(self):
        self.cfg = json.loads((AI / "config/hybrid_v7.json").read_text(encoding="utf-8"))
        self.cfg["food_mouth_crop_enabled"] = False

    def pipeline(self, *, phone_box=None, phone_score=.8, sleep_score=None, pose=None,
                 face=None, food_box=None, food_class="apple", food_owner="p"):
        from src.hybrid_v7 import spatial_v7
        pbox = [0, 0, 200, 400]
        detections = []
        if phone_score is not None:
            detections.append({"class_name": "using_phone", "confidence": phone_score,
                               "bbox_xyxy": pbox})
        if sleep_score is not None:
            detections.append({"class_name": "sleeping", "confidence": sleep_score,
                               "bbox_xyxy": pbox})
        phones = ([{"detection_id": "ph", "bbox_xyxy": phone_box, "confidence": .8}]
                  if phone_box else [])
        foods = ([{"detection_id": "food", "bbox_xyxy": food_box, "class_name": food_class,
                   "confidence": .8}] if food_box else [])
        obj = {"phone_detections": phones, "food_detections": foods,
               "food_person_associations": ([{"food_detection_id": "food", "person_detection_id": food_owner}]
                                            if foods else []),
               "v6_phone_person_associations": ([{
                   "phone_detection_id": "ph", "person_detection_id": "p",
                   "status": "ASSOCIATED_GEOMETRICALLY", "candidate_person_ids": ["p"]
               }] if phones else [])}
        base = {"people": [{"person_ref": "p", "bbox_xyxy": pbox,
                            "person_bbox_xyxy_clipped": pbox, "yolo_detections": detections,
                            "evidence": {"face": face or {}, "pose_landmarks": pose or {}},
                            "behaviors": {b: {"state": "unknown", "evidence": {}}
                                          for b in ("sleeping", "using_phone", "eating", "turning_back")}}],
                "phone_detections": phones, "food_detections": foods}
        with patch("src.hybrid_v7.spatial_v3", return_value=deepcopy(base)):
            return spatial_v7(None, detections, obj, None, self.cfg, None, None, None)["people"][0]["behaviors"]

    def test_final_v7_keeps_desk_phone_suspected(self):
        result = self.pipeline(phone_box=[50, 300, 110, 320])
        self.assertEqual(result["using_phone"]["state"], "suspected")
        self.assertEqual(result["using_phone"]["decision_reason"],
                         "flat_low_phone_without_nearby_wrist_may_be_resting_on_surface")
        self.assertIsNone(result["using_phone"]["bounding_box"])

    def test_final_v7_wrist_interaction_can_confirm_phone(self):
        result = self.pipeline(phone_box=[50, 300, 110, 320],
                               pose={"left_wrist": {"x": 70, "y": 310, "visibility": .99}})
        self.assertEqual(result["using_phone"]["state"], "detected")

    def test_final_v7_phone_near_face_does_not_require_wrist(self):
        result = self.pipeline(phone_box=[70, 30, 90, 70])
        self.assertEqual(result["using_phone"]["state"], "detected")

    def test_final_v7_sleep_conflict_does_not_confirm_phone_without_interaction(self):
        result = self.pipeline(phone_box=[70, 160, 90, 200], phone_score=.35, sleep_score=.9)
        self.assertEqual(result["using_phone"]["state"], "suspected")
        self.assertTrue(result["using_phone"]["evidence"]["sleeping_phone_conflict"])
        self.assertNotEqual(result["sleeping"]["state"], "detected")

    def test_final_v7_preserves_geometry_check(self):
        result = self.pipeline(phone_box=[20, 80, 160, 250])
        self.assertEqual(result["using_phone"]["state"], "suspected")
        self.assertEqual(result["using_phone"]["evidence"]["reason_code"], "PHONE_OBJECT_GEOMETRY_IMPLAUSIBLE")

    def test_final_v7_no_behavior_candidate_does_not_confirm_phone_object(self):
        result = self.pipeline(phone_box=[70, 30, 90, 70], phone_score=None)
        self.assertEqual(result["using_phone"]["state"], "unknown")

    def test_final_v7_sleep_threshold_without_face_or_pose(self):
        for score, expected in ((None, "not_detected"), (.74, "suspected"), (.75, "detected")):
            with self.subTest(score=score):
                result = self.pipeline(phone_score=None, sleep_score=score)
                self.assertEqual(result["sleeping"]["state"], expected)

    def test_final_v7_turning_requires_classroom_rear_calibration(self):
        result = self.pipeline(phone_score=None)
        self.assertEqual(result["turning_back"]["state"], "unknown")
        self.assertEqual(result["turning_back"]["decision_reason"], "rear_direction_uncalibrated")

    def test_final_v7_eating_requires_food_near_visible_mouth(self):
        cases = [
            ({"food_box": [90, 70, 110, 90], "face": {"mouth_center_xy_frame": [100, 80]}}, "detected"),
            ({"food_box": [90, 70, 110, 90]}, "suspected"),
            ({"food_box": [90, 300, 110, 320], "face": {"mouth_center_xy_frame": [100, 80]}}, "not_detected"),
            ({"food_box": [90, 70, 110, 90], "face": {"mouth_center_xy_frame": [100, 80]}, "food_owner": "other"}, "unknown"),
            ({"food_box": [90, 70, 110, 90], "face": {"mouth_center_xy_frame": [100, 80]}, "food_class": "cup"}, "unknown"),
        ]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                result = self.pipeline(phone_score=None, **inputs)
                self.assertEqual(result["eating"]["state"], expected)

    def test_exported_prepare_uses_v7_local_rules(self):
        from src import hybrid_v7
        self.assertIs(hybrid_v7.prepare_objects_v7, hybrid_v7.prepare_objects_v6_1)

    def test_crop_origin_cannot_resolve_overlapping_people(self):
        from src.hybrid_v7 import prepare_objects_v7
        people = [{"detection_id": "a", "bbox_xyxy": [0, 0, 100, 200]},
                  {"detection_id": "b", "bbox_xyxy": [50, 0, 150, 200]}]
        base = {"person_detections": people, "phone_detections": [], "food_detections": []}
        crop = [{"detection_id": "ph", "class_name": "cell phone", "confidence": .8,
                 "bbox_xyxy": [60, 100, 80, 120], "source": "person_crop_coco", "person_crop_ref": "a"},
                {"detection_id": "food", "class_name": "apple", "confidence": .8,
                 "bbox_xyxy": [60, 30, 80, 50], "source": "person_crop_coco", "person_crop_ref": "a"}]
        objects = type("Objects", (), {"food_classes": {"apple"}})()
        with patch("src.hybrid_v7.prepare_objects_v6", return_value=deepcopy(base)), \
             patch("src.hybrid_v7._crop_person_object_proposals", return_value=crop):
            result = prepare_objects_v7(None, objects, config=self.cfg)
        for key in ("v6_phone_person_associations", "food_person_associations"):
            self.assertIsNone(result[key][0]["person_detection_id"])
            self.assertEqual(set(result[key][0]["candidate_person_ids"]), {"a", "b"})

    def test_mouth_crop_food_requires_unique_person_owner(self):
        from src.hybrid_v7 import spatial_v7
        for overlapping in (False, True):
            with self.subTest(overlapping=overlapping):
                boxes = [[0, 0, 100, 200]]
                if overlapping:
                    boxes.append([40, 0, 140, 200])
                people = [{"person_ref": str(i), "bbox_xyxy": box,
                           "evidence": {"face": {"mouth_center_xy_frame": [60, 35]}},
                           "behaviors": {"sleeping": {"state": "unknown"}, "eating": {"evidence": {}}}}
                          for i, box in enumerate(boxes)]
                food = {"detection_id": "mouth_food_0", "class_name": "apple", "confidence": .8,
                        "bbox_xyxy": [50, 25, 70, 45], "person_ref": "0"}
                base = {"people": people, "food_detections": [], "v6_1_food_associations": []}
                with patch("src.hybrid_v7.spatial_v6_1", return_value=base), \
                     patch("src.hybrid_v7.mouth_focused_food_proposals_v7", return_value=([food], {})):
                    out = spatial_v7(None, [], {}, None, self.cfg, None, None, None)
                expected = "unknown" if overlapping else "detected"
                self.assertEqual(out["people"][0]["behaviors"]["eating"]["state"], expected)
                if overlapping:
                    self.assertEqual(out["people"][1]["behaviors"]["eating"]["state"], "unknown")

    def test_ambiguous_phone_without_owner_demotes_sleep(self):
        from src.hybrid_v7 import apply_ambiguous_sleep_phone_conflicts_v7
        person = {"person_ref": "a", "bbox_xyxy": [0, 0, 100, 200], "behaviors": {
            "sleeping": {"state": "detected", "evidence": {"sleeping_yolo_box": [0, 0, 100, 200]}}}}
        out = {"people": [person], "phone_detections": [
            {"detection_id": "ph", "bbox_xyxy": [60, 100, 80, 120], "confidence": .8}],
            "v6_1_phone_associations": [{"phone_detection_id": "ph", "person_detection_id": None,
                                        "status": "UNCERTAIN", "candidate_person_ids": ["a", "b"]}]}
        self.assertEqual(len(apply_ambiguous_sleep_phone_conflicts_v7(out, self.cfg)), 1)
        self.assertEqual(person["behaviors"]["sleeping"]["state"], "suspected")


if __name__=="__main__":unittest.main()
