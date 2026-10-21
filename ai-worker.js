import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

env.allowLocalModels = false;
// A larger *real* local model. All inference remains inside the browser.
const MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct';
let generator = null;
let loading = null;

async function load() {
  if (generator) return generator;
  if (loading) return loading;
  loading = pipeline('text-generation', MODEL, {
    device: 'wasm', dtype: 'q4',
    progress_callback: p => {
      if (p.status === 'progress' && p.total) {
        const n = Number.isFinite(p.progress) ? p.progress : 100 * p.loaded / p.total;
        self.postMessage({type:'progress', text:`Downloading local AI: ${Math.round(n)}%`});
      } else if (p.status === 'initiate' || p.status === 'download') {
        self.postMessage({type:'progress',text:'Preparing model files…'});
      }
    }
  }).then(p => { generator=p; self.postMessage({type:'ready'}); return p; })
    .catch(e => {loading=null; throw e;});
  return loading;
}

// Small deterministic calculator: a utility, not a pretend language-model answer.
function calculation(query) {
  const m = query.trim().replace(/[?=]+$/,'').trim().match(/^(?:what is|what's|calculate|compute|solve)?\s*(-?\d+(?:\.\d+)?)\s*(\+|\-|\*|×|x|\/|÷)\s*(-?\d+(?:\.\d+)?)$/i);
  if (!m) return null;
  const a=Number(m[1]), b=Number(m[3]);
  let result;
  switch(m[2]) {case '+':result=a+b;break;case '-':result=a-b;break;case '*':case '×':case 'x':result=a*b;break;default:if(b===0)return 'You cannot divide by zero.';result=a/b;}
  return Number.isFinite(result) ? String(Number(result.toPrecision(12))) : 'That calculation is too large.';
}

function lastMessages(history) {
  const clean=(Array.isArray(history)?history:[])
    .filter(m=>m && (m.role==='user'||m.role==='assistant') && typeof m.content==='string' && m.content.trim())
    .slice(-5).map(m=>({role:m.role,content:m.content.slice(0,600)}));
  return [
    {role:'system',content:'You are Sherry, a friendly AI. Answer the latest user message directly and naturally. Be concise. Do not repeat generic greetings or introduce yourself unless asked.'},
    ...clean
  ];
}

self.onmessage=async event=>{
  const {type,id,history}=event.data||{};
  try {
    if (type==='load') {await load();return;}
    if(type!=='ask')return;
    const recent = (Array.isArray(history)?history:[]).filter(m=>m?.role==='user');
    const last=recent.at(-1)?.content||'';
    const exact=calculation(last);
    if(exact!==null) {self.postMessage({type:'answer',id,answer:exact});return;}
    const engine=await load();
    self.postMessage({type:'progress',text:'Sherry is thinking…'});
    const result=await engine(lastMessages(history),{
      max_new_tokens:120,
      do_sample:false,
      return_full_text:false,
      repetition_penalty:1.12
    });
    let answer=result?.[0]?.generated_text;
    if(Array.isArray(answer))answer=answer.filter(x=>x?.role==='assistant').at(-1)?.content ?? answer.at(-1)?.content;
    if(typeof answer!=='string')answer=String(answer??'');
    answer=answer.trim().replace(/^(assistant\s*:\s*)/i,'');
    self.postMessage({type:'answer',id,answer:answer||'I couldn’t generate an answer. Try asking differently.'});
  }catch(e){self.postMessage({type:'error',id,error:String(e?.message||e)});}
};
