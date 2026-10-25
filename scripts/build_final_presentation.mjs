/** Editable defense deck. Uses the Codex bundled @oai/artifact-tool runtime. */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.ARTIFACT_RUNTIME_ROOT ?? path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies');
process.env.RUNTIME_NODE_MODULES ??= path.join(runtime,'node/node_modules');
const skill=process.env.PRESENTATIONS_SKILL_DIR ?? path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation,applyPresentationChartFont,resolvePresentationFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const font=resolvePresentationFont({fontFamily:'Arial'});
const content=JSON.parse(await fs.readFile(path.join(root,'output/final-presentation/content.json'),'utf8'));
const build=path.join(root,'.presentation-build');
const dest=path.join(root,'output/final-presentation');
await fs.mkdir(build,{recursive:true});await fs.mkdir(dest,{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const colors=['#215D91','#C45B24','#287C62'];
const tables=[],charts=[];
function text(slide,value,x,y,w,h,size=29,color='#172A3A',bold=false){
 const sh=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=value;sh.text.style={typeface:font,fontSize:size,color,bold,verticalAlignment:'top'};
 return sh;
}
// Editable equation layout: fractions, summation limits and true positioned subscripts.
function roughnessFormula(slide, order, x, y){
 text(slide,`R${order===1?'₁':'₂'}(x) =`,x,y+25,170,45,32);
 text(slide,'1',x+203,y+4,48,35,27);
 slide.shapes.add({geometry:'rect',position:{left:x+173,top:y+40,width:90,height:1.5},fill:'#172A3A',line:{fill:'none',width:0}});
 text(slide,`H − ${order}`,x+169,y+44,110,38,27);
 text(slide,`H − ${order}`,x+283,y-8,80,29,18);
 text(slide,'∑',x+288,y+12,72,54,41);
 text(slide,'i = 1',x+288,y+60,80,26,18);
 let pos=x+364;
 const token=(str,width)=>{text(slide,str,pos,y+25,width,48,32);pos+=width;};
 const term=(sub)=>{text(slide,'x',pos,y+25,27,45,32);text(slide,sub,pos+19,y+47,55,28,18);pos+=sub.length>1?66:46;};
 const bar=()=>{slide.shapes.add({geometry:'rect',position:{left:pos+8,top:y+31,width:1.5,height:32},fill:'#172A3A',line:{fill:'none',width:0}});pos+=17;};
 bar();term(`i+${order}`);token('−',36);
 if(order===2){token('2',23);term('i+1');token('+',36);}
 term('i');bar();
}
for(let i=0;i<content.slides.length;i++){
 const d=content.slides[i],s=deck.slides.add();s.background.fill=d.kind==='cover'?'#132D46':'#FFFFFF';
 if(d.kind==='cover'){
  text(s,d.title,76,100,1100,205,62,'#FFFFFF',true);
  text(s,d.subtitle,80,330,1000,110,34,'#D1E4F3');
  text(s,d.body.join('\n'),80,540,1100,100,25,'#FFFFFF');
 }else{
  text(s,d.title,64,42,1152,108,44,'#172A3A',true);
  if(d.roughness_formula){roughnessFormula(s,d.roughness_formula.order,76,d.roughness_formula.top);text(s,d.roughness_formula.definition,835,d.roughness_formula.top+8,365,78,21,'#43596C');}
  if(d.ratio_line)text(s,d.ratio_line,76,d.ratio_top,1120,30,21,'#43596C');
  if(d.intro)text(s,d.intro,76,d.intro_top??150,1120,d.intro_height??100,d.intro_font_size??23,'#43596C');
  if(d.kind==='text'){
   let y=180;
   for(const b of d.body){text(s,b,78,y,1100,86,d.body_font_size??29);y+=100;}
  }else if(d.kind==='figure'){
   s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,d.figure))),contentType:'image/png',alt:d.alt,fit:'contain',
    position:{left:64,top:d.figure_top??159,width:1152,height:d.figure_height??420}});
   if(d.body)for(let j=0;j<d.body.length;j++)text(s,d.body[j],78,(d.body_top??450)+j*(d.body_gap??58),1120,54,d.body_font_size??27);
  }else if(d.kind==='table'){
   const vals=[d.headers,...d.rows];
   const n=d.headers.length;
   const widths=d.column_widths ?? (n===2?[310,826]:n===3?[410,371,355]:[440,220,238,238]);
   const height=d.table_height ?? Math.min(410,vals.length*70);
   const table=s.tables.add({rows:vals.length,columns:n,left:72,top:d.table_top??176,width:1136,height,values:vals,columnWidths:widths});
   table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#D5E0E8',width:1});
   for(let r=0;r<vals.length;r++)for(let c=0;c<n;c++){
    const cell=table.getCell(r,c);cell.fill=r===0?'#E5EFF6':'#FFFFFF';
    cell.text.style={typeface:font,fontSize:d.table_font_size??24,color:'#172A3A',bold:r===0};
   }
   if(d.body)for(let j=0;j<d.body.length;j++)text(s,d.body[j],78,(d.table_top??176)+height+28+j*(d.body_gap??52),1120,d.body_height??48,d.body_font_size??27);
   tables.push(i+1);
  }else if(d.kind==='chart'){
   const cfg=d.chart;
   if(cfg.caption)text(s,cfg.caption,100,cfg.caption_top??154,1110,cfg.caption_height??32,21,'#43596C');
   const ch=s.charts.add('bar',{position:{left:100,top:cfg.plot_top??(cfg.caption?195:165),width:1060,height:cfg.plot_height??(cfg.caption?365:395)},categories:cfg.categories,
    series:cfg.series.map((v,j)=>({...v,values:v.values.map(x=>Number(x.toPrecision(8))),fill:colors[j]})),barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:24,fill:'#172A3A'}},
    xAxis:{textStyle:{fontSize:cfg.category_size??24,fill:'#172A3A'},majorGridlines:null},
    yAxis:{min:0,...(cfg.max?{max:cfg.max}:{}),numberFormatCode:cfg.format,title:{text:cfg.axis,textStyle:{fontSize:23,fill:'#172A3A'}},textStyle:{fontSize:23,fill:'#172A3A'},majorGridlines:{fill:'#D5E0E8',width:1}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:cfg.format,textStyle:{fontSize:cfg.label_size??23,fill:'#172A3A'}}});
   applyPresentationChartFont(ch,{fontFamily:font});charts.push(i+1);
  }
  if(d.bottom)text(s,d.bottom,76,d.bottom_top??596,1120,d.bottom_height??74,d.bottom_font_size??25,'#43596C');
  text(s,`${i+1}`,1150,683,95,24,17,'#60788B');
 }
 s.speakerNotes.text=`${d.backup?'Backup only':`Time budget: ${d.seconds} seconds`}\n\n${d.notes}`;
}
await fs.mkdir(path.join(build,'render'),{recursive:true});
for(let i=0;i<deck.slides.items.length;i++){
 const blob=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,'render',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidate);
const final=path.join(dest,'defense.pptx');
try{await fs.rename(final,path.join(build,'previous-defense.pptx'));}catch(e){if(e.code!=='ENOENT')throw e;}
const receipt=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,
 pythonExecutable:path.join(runtime,'python/bin/python3'),
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(v=>['--require-native-table-slide',String(v)])],
 explicitTotalSlideCount:content.slides.length,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:charts,
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(build,`validation-${Date.now()}.json`)});
console.log(JSON.stringify({slides:content.slides.length,main_minutes:content.main_minutes,file:final,receipt}));
