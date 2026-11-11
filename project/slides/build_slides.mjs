// Plain academic presentation of the independently trained fresh-view experiment.
// Uses the Codex bundled @oai/artifact-tool. No decorative template or image assets.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath,pathToFileURL } from 'node:url';
import { Presentation,PresentationFile } from '@oai/artifact-tool';
const dir=path.dirname(fileURLToPath(import.meta.url));
const project=process.env.TOMATO_PROJECT_ROOT ? path.resolve(process.env.TOMATO_PROJECT_ROOT) : path.dirname(dir);
const data=JSON.parse(await fs.readFile(path.join(project,'slides/slide_data.json'),'utf8'));
const pres=Presentation.create({slideSize:{width:1280,height:720}});
const refs='PlantVillage: https://doi.org/10.3389/fpls.2016.01419; https://github.com/spMohanty/PlantVillage-Dataset\nCropped-PlantDoc: https://doi.org/10.1145/3371158.3371196; https://github.com/pratikkayal/PlantDoc-Dataset (CC BY 4.0)\nMobileNetV3: https://arxiv.org/abs/1905.02244\nNew experiment: results/fresh_views/protocol.json, events.jsonl, per_seed_metrics.csv, summary.csv and training_audit.csv. Legacy results remain separately in results/.';
function text(slide,value,x,y,w,h,size=28,bold=false){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={typeface:'Arial',fontSize:size,bold,color:'#000000'};return s;
}
function slide(title,notes=''){
 const s=pres.slides.add();s.background.fill='#FFFFFF';text(s,title,65,42,1150,85,44,true);
 text(s,String(pres.slides.items.length),1180,670,50,24,18);s.speakerNotes.text=notes+'\n\nSources:\n'+refs;return s;
}
function lines(s,values,y=160,size=29,gap=100){values.forEach((v,i)=>text(s,v,80,y+i*gap,1120,gap-12,size));}
function table(s,values,y=165,h=350,size=28,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:75,top:y,width:1130,height:h,values,columnWidths:widths});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:'#FFFFFF',textStyle:{typeface:'Arial',fontSize:size,color:'#000000'},margins:{left:10,right:10,top:6,bottom:6}});
 t.borders.assign({fill:'#888888',width:.6,style:'solid'});
 for(let c=0;c<values[0].length;c++){t.getCell(0,c).fill='#EEEEEE';t.getCell(0,c).text.style={typeface:'Arial',fontSize:size,bold:true,color:'#000000'};}
 return t;
}
async function picture(s,filename,x,y,w,h,alt){
 const b=await fs.readFile(path.join(project,filename));
 s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:filename.endsWith('.png')?'image/png':'image/jpeg',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
let s=slide('Plant disease classification','Question: how much clean-to-field performance is lost, and can crop/color augmentation reduce it? The main comparison now consists of six independently trained models. The legacy two-fixed-view single-seed result is retained as a reference.');
text(s,'Fresh views and three paired seeds',65,195,1140,110,48,true);
text(s,'Can crop and color augmentation improve field performance?',65,365,1140,100,30);
text(s,'COMP 664 · Topic 2 · Fall 2026',65,535,1140,50,26);

s=slide('Data and split integrity','The source split is unchanged. Source validation only uses documented training leaf IDs and supports eight classes, while both tests support nine. 4,280 training and 1,060 clean-test images lack documented leaf IDs. Exact-byte checks cannot rule out all same-leaf identities or near-duplicates.');
table(s,[['Dataset','Purpose','Images'],['PlantVillage','Training','12,036'],['PlantVillage','Validation','1,364'],['PlantVillage','Clean test','3,351'],['PlantDoc','External test','742']],160,310,29,[340,530,260]);
text(s,'Nine test classes; mosaic virus has no validation examples.\nPlantDoc never enters optimization or checkpoint selection.',80,520,1120,105,28);

s=slide('Six newly trained models','Three fixed seeds: 664, 665, 666. Both conditions start from identical weights within seed and use the same shuffled indices. No field-based run selection. Training uses deterministic CUDA algorithms with TF32 disabled, PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128.');
lines(s,['Three seeds × ordinary / strong augmentation × eight epochs.', 'Fresh per-image transformation draws every epoch.\nOnly deterministic RGB resize is cached.', 'Same ImageNet MobileNetV3-Small, trainable blocks, AdamW,\nbatch size 96 and 1,008 steps per model.', 'All six checkpoints selected on source validation\nbefore either test set is scored.'],155,29,118);

s=slide('Training and validation history','Means over the three seeds; shading is sample SD. Supported validation macro-F1 covers eight classes. Each model selects its first maximum independently. Training loss is cross-entropy. The training logs record all 48 epochs, not merely saved predictions.');
await picture(s,'report/figures/fresh_history.png',65,150,1150,365,'Mean and sample SD of loss and validation macro-F1 across the three seeds');
text(s,'48 completed epochs · 6,048 optimizer steps · 577,728 image presentations',80,570,1120,70,28);

s=slide('Mean test results across three seeds','Each number is mean ± sample SD in percent. The three models are evaluated on the same images, so seed replication does not triple test-set sample size. All selected checkpoints are included. Strong augmentation improves field metrics on average, while source metrics fall modestly.');
table(s,[['Metric (%)','Ordinary','Crop + color'],...data.overall],165,330,29,[500,315,315]);
text(s,'Mean ± sample SD over seeds 664, 665 and 666.\nSame 3,351 clean images and 742 field images for every run.',80,545,1120,95,28);

s=slide('Every field result, with selected epochs','Epochs are chosen from source-validation macro-F1, never field scores. This slide displays every one of the six new models. There is no best-field-seed selection or ensemble reported as a substitute for individual results.');
table(s,[['Seed','Training','Epoch','Accuracy (%)','Macro-F1 (%)'],...data.seeds],150,420,27,[110,330,130,280,280]);
text(s,'Selected epochs use PlantVillage validation only.',80,605,1120,45,28);

s=slide('Paired field changes vary by seed','Each row is strong minus ordinary within the same seed, in percentage points. Two seeds improve on both metrics and one declines slightly. Three paired observations are insufficient for a universal benefit or equivalence claim. Per-seed paired image bootstrap intervals remain in the data package and condition on each fitted pair.');
table(s,[['Seed','Accuracy change (pp)','Macro-F1 change (pp)'],...data.paired],165,315,28,[330,400,400]);
text(s,'Two seed pairs improve; one declines slightly.\nThe mean gain is modest and not consistent across all seeds.',80,535,1120,100,29);

s=slide('Mean per-class field recall','Three-seed mean recall in percent. Delta is strong minus ordinary in percentage points. The two spider-mite images are related multi-panel illustrations; seeds do not create additional independent examples. Per-class clean-minus-field gaps and F1 are in the report and CSVs.');
table(s,[['Class','Field n','Ordinary (%)','Strong (%)','Change (pp)'],...data.recall],145,470,25,[380,110,210,210,220]);

s=slide('Field confusion matrices for seed 664','The illustrative seed 664 was specified before viewing new test metrics. Cell text is a count, shading is within-row proportion. Both models overpredict late blight; averaging this pattern across seeds does not remove class imbalance. The all-seed prediction counts are in results/fresh_views/predicted_class_counts.csv.');
await picture(s,'report/figures/fresh_confusions.png',65,140,1150,480,'Actual new seed-664 field confusion matrices');
text(s,'Seed 664 was fixed for illustration before field scoring.',80,630,1120,40,27);

s=slide('Fixed demonstration photographs','Both labels are early blight. These two examples were fixed during legacy analysis before new training and were not reselected based on fresh-model performance. Predictions below are from the new seed-664 checkpoints. Exact source paths: examples/credits.csv, field.jpg and field_second.jpg.');
await picture(s,'examples/field.jpg',80,155,515,290,'First fixed early-blight photograph from PlantDoc');
await picture(s,'examples/field_second.jpg',680,155,515,290,'Second fixed early-blight photograph from PlantDoc');
for(let i=0;i<2;i++){const e=data.examples[i];text(s,'Label: Early blight\nOrdinary: '+e.ordinary+'\nStrong: '+e.strong,i===0?90:690,485,500,140,28);}

s=slide('Conclusion and remaining limitations','Observed conclusion: '+data.interpretation.conclusion+' The follow-up was designed after the legacy test results were inspected, so this is exploratory evidence on an already examined benchmark. Field labels never enter optimization or checkpoint selection. The old CPU single-seed comparison is not a pure ablation of fresh views because RNG and environment also changed.');
lines(s,['Mean field accuracy and macro-F1 improve modestly;\nthe benefit varies across the three seeds.', 'Remaining limits: missing validation class and leaf IDs,\nnear-duplicates, only two spider-mite field images.', 'A stronger next study needs independent field labels\nand a predefined target-development/test separation.'],160,30,145);

s=slide('Notebook and reproducibility evidence','The executed Notebook independently recomputes all seed results, class metrics and paired intervals, then loads real new checkpoints for example inference. Its default run does not repeat training. Fresh actual training evidence: protocol.json, events.jsonl, seed completion records, all_history.csv, checkpoint hashes in selection_locked.json and aggregate_verification.json. Original weights and predictions remain separate.');
lines(s,['Open Tomato_Project.ipynb for code, figures and saved outputs.', 'New method: experiment_fresh.py\nAggregate and audit: summarize_fresh.py', 'Six checkpoints, complete training histories, all test probabilities\nand the unchanged original experiment are included.'],160,29,145);

const build=process.env.SLIDES_BUILD_DIR ? path.resolve(process.env.SLIDES_BUILD_DIR) : path.join(project,'.build','slides',String(Date.now()));
await fs.mkdir(build,{recursive:true});
await fs.mkdir(path.join(build,'deliverable'),{recursive:true});
await fs.writeFile(path.join(project,'slides/presentation.json'),JSON.stringify(pres.toProto(),null,2));
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
if(process.env.CODEX_PRESENTATION_SKILL){
 const skill=process.env.CODEX_PRESENTATION_SKILL;
 const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
 const result=await finalizePresentation({workspaceDir:path.resolve(project,'../../..'),candidatePath:candidate,finalPath:path.join(build,'deliverable','final.pptx'),pythonExecutable:process.env.CODEX_RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,5,6,7,8].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[2,5,6,7,8],fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
 console.log(JSON.stringify({finalPath:result.finalPath,slides:12,integrity:result.packageIntegrity.findingCount,layout:result.presentationLayout.findingCount}));
}else{console.log(JSON.stringify({candidate,slides:12}));}
