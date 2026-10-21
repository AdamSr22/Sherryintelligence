import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

env.allowLocalModels = false;
const MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct';
let generator = null;
let loading = null;

async function load() {
  if (generator) return generator;
  if (loading) return loading;
  loading = pipeline('text-generation', MODEL, {
    device: 'wasm', dtype: 'q8',
    progress_callback: (p) => {
      if (p.status === 'progress' && p.total) {
        self.postMessage({type:'progress', text:`Downloading AI: ${Math.round(p.progress ?? p.loaded / p.total * 100)}%`});
      } else if (p.status === 'initiate' || p.status === 'download') {
        self.postMessage({type:'progress', text:'Preparing local AI files…'});
      }
    }
  }).then(p => {
    generator = p;
    self.postMessage({type:'ready'});
    return p;
  }).catch(e => { loading=null; throw e; });
  return loading;
}

self.onmessage = async (event) => {
  const {type, id, history} = event.data;
  try {
    if (type === 'load') { await load(); return; }
    if (type === 'ask') {
      const engine = await load();
      self.postMessage({type:'progress', text:'Sherry is thinking…'});
      const messages = [
        {role:'system',content:'You are Sherry, a friendly and helpful little AI assistant inside SherryIntelligence. Answer briefly and clearly. Admit uncertainty. You have no live internet access. You run entirely on this phone.'},
        ...history.slice(-8)
      ];
      const result = await engine(messages, {
        max_new_tokens: 100, do_sample:false, return_full_text:false
      });
      let answer = result?.[0]?.generated_text;
      if (Array.isArray(answer)) answer = answer.at(-1)?.content;
      if (typeof answer !== 'string') answer = JSON.stringify(answer ?? '');
      answer = answer.trim().replace(/^assistant\s*:\s*/i, '');
      self.postMessage({type:'answer', id, answer:answer || 'Sorry, I could not make a response. Try again!'});
    }
  } catch (e) {
    self.postMessage({type:'error', id, error:String(e?.message || e)});
  }
};
