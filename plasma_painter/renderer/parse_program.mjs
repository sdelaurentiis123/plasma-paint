// Parsing only: never execute candidate code in this process.
import {parse} from 'acorn';
let source='';
for await (const chunk of process.stdin) source+=chunk;
try {
  const comments=[];
  const ast=parse(source,{ecmaVersion:2022,sourceType:'module',allowReturnOutsideFunction:true,
    onComment:(_block,_text,start,end)=>comments.push([start,end])});
  let normalized=source,format='body';
  if(process.argv[2]==='body' && ast.body.length===1 && ast.body[0].type==='FunctionDeclaration') {
    const f=ast.body[0];
    if(f.async || f.generator || f.params.map(p=>p.name).join(',')!=='frameFeatures,time,persistentState')
      throw new Error('Painter function must take exactly (frameFeatures,time,persistentState) and be synchronous');
    normalized=source.slice(f.body.start+1,f.body.end-1);
    format='function_body_extracted';
  }
  let clean=source;
  for(const [start,end] of comments.reverse()) clean=clean.slice(0,start)+clean.slice(start,end).replace(/[^\n\r]/g,' ')+clean.slice(end);
  process.stdout.write(JSON.stringify({clean,normalized,format}));
} catch(error) {process.stderr.write(error.message);process.exitCode=1;}
