import {pipeline,env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';
env.allowLocalModels=false;
let pipe=null;
self.onmessage=async ({data})=>{
  try {
    if(data.type==='load') {
      if(!pipe) pipe=await pipeline('text-generation','HuggingFaceTB/SmolLM2-360M-Instruct',{
        device:'wasm',dtype:'q4',progress_callback:p=>{
          if(p.status==='progress' && p.total) self.postMessage({type:'progress',message:'Download: '+Math.round((p.progress??(100*p.loaded/p.total)))+'%'});
          else if(p.status==='initiate'||p.status==='download') self.postMessage({type:'progress',message:'Preparing AI files…'});
        }
      });
      self.postMessage({type:'ready'});
    }
    if(data.type==='ask') {
      if(!pipe)throw Error('Load AI first');
      const question=String(data.question||'').slice(0,250);
      const messages=[{role:'system',content:'You are Sherry. Answer the question directly, briefly and accurately. If unsure, say so.'},{role:'user',content:question}];
      const res=await pipe(messages,{max_new_tokens:64,do_sample:false,return_full_text:false});
      let output=res?.[0]?.generated_text;
      if(Array.isArray(output))output=output.filter(x=>x.role==='assistant').at(-1)?.content??'';
      self.postMessage({type:'answer',message:String(output??'').trim()||'(No reply generated)'});
    }
  }catch(err){self.postMessage({type:'error',message:String(err?.message||err)});}
};
