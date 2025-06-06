import nltk
from nltk.tokenize import word_tokenize
from bert_score import score
import pandas as pd

nltk.download('punkt')

# 三筆資料
data = [
    {
        "id": "sub007",
        "model_answer": """Radiography of Lumbar Spine (Flexion View)

This flexion view of the lumbar spine shows several notable findings:
- Vertebral Body Size and Shape: The vertebral bodies appear to be of normal size with no signs of compression fractures or spondylolisthesis.
- Intervertebral Discs: The intervertebral discs appear to have a normal height without evidence of bulging, herniation, or disc degeneration.
- Spinal Canal: There is no narrowing of the spinal canal which would suggest spinal stenosis.
- Spondylolysis/Fracture: No evidence of spondylolysis or fracture at any level visible on this view.
- Bone Density: The bone density appears to be within normal limits.
""",
        "ground_truth": """Radiography of Chest (P-A View, Standing), Pelvis (A-P View Including Upper Femur), and L-S Spine (Flexion + Extension Lateral View) Show:
- Cardiomegaly.
- Tortuousity of thoracic aorta and calcified aortic knob.
- Clear costophrenic angles bilaterally.
- Increased interstitial infiltration in both lungs.
- Elevated right hemidiaphragm.
- S/P ORIF of the left humerus.
- Degenerative spine with spur formation.
- Compression fracture of T12 S/P bone cement vertebroplasty.
- Compression fracture of L2.
- Minimal retrolisthesis at L2-3.
- Old fracture of the right pubic bone.
"""
    },
    {
        "id": "sub010",
        "model_answer": """Radiography of Lumbar Spine (Lateral View)

- Vertebral Body Alignment: The lumbar vertebrae appear to be aligned without evidence of significant vertebral body fractures or dislocations.
- Intervertebral Discs: There is no obvious evidence of herniated discs, disc bulging, or narrowing of intervertebral spaces.
- Spondylolisthesis/Slippage: No signs of spondylolisthesis or slippage of any vertebra into the adjacent one.
- Bone Density and Structure: The bones exhibit normal density with intact cortices; there is no evidence of osteoporosis or other bone abnormalities such as osteolytic lesions or extensive bone absorption.
- Surgical Interventions: Several metal components suggestive of spinal fusion procedures can be identified. These include screws, rods, and cages placed bilaterally across multiple lumbar vertebrae, indicating surgical intervention aimed at stabilizing the spine.
- Soft Tissues: Soft tissues around the vertebrae and surrounding structures do not demonstrate abnormal thickening, calcifications, or masses.
""",
        "ground_truth": """Radiography of L-S Spine (A-P View, Lateral View) Show:
- Degenerative change of lumbar spine with spur formation.
- S/P transpedicular screw fixation at L2-S1.
- S/P disc prosthesis at L3-S1.
- Disc space narrowing at L1-2 level.
"""
    },
    {
        "id": "sub013",
        "model_answer": """Radiography of Lumbar Spine (Anterior View)

- Vertebral Body Compression: There is significant compression and narrowing of the vertebral body at L2-L3, indicating degenerative changes.
- Spondylolisthesis: A grade III spondylolisthesis is present at L5-S1 with posterior slip on the left side, which may suggest mechanical irritation to adjacent structures.
- Adjacent Discs: The intervertebral discs between L2-L3 and L5-S1 appear to be herniated or bulging, suggesting potential nerve root compression.
- Bony Anatomy: There appears to be a hardware implant present within the region, likely representing an anterior spinal fusion device used post-surgery. This could explain some of the bone anatomy alterations seen.
""",
        "ground_truth": """Radiography of Lumbar Spine (A-P and Lateral Views) Show:
- Degenerative spine with spur formation.
- S/P laminectomy with transpedicle screw placement at L2-L5.
- S/P bone cement vertebroplasty at L4.
- S/P drainage tube placement.
"""
    }
]

# 儲存結果
results = []

for item in data:
    ref_tokens = set(word_tokenize(item['ground_truth'].lower()))
    cand_tokens = set(word_tokenize(item['model_answer'].lower()))
    recall = len(ref_tokens & cand_tokens) / len(ref_tokens) if ref_tokens else 0.0

    P, R, F1 = score([item["model_answer"]], [item["ground_truth"]], lang='en', verbose=False)
    
    results.append({
        "ID": item["id"],
        "Token-Level Recall": round(recall, 4),
        "BERTScore_P": round(P[0].item(), 4),
        "BERTScore_R": round(R[0].item(), 4),
        "BERTScore_F1": round(F1[0].item(), 4)
    })

df = pd.DataFrame(results)
print(df)