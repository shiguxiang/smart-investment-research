import { useState, useEffect, useRef, useCallback } from 'react';
import { Zap, Cpu, Shield, Loader2 } from 'lucide-react';
import Sidebar from './components/Sidebar';
import AgentPipeline from './components/AgentPipeline';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import FileUpload from './components/FileUpload';
import { sendMessage, checkHealth } from './api';

function now() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [files, setFiles] = useState([]);
  const [subject, setSubject] = useState('');
  const [sessionId] = useState(() => Math.random().toString(36).slice(2, 10));
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('connecting');
  const [showFiles, setShowFiles] = useState(false);
  const [activeStep, setActiveStep] = useState('');
  const [error, setError] = useState('');

  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  // 健康检查
  useEffect(() => {
    checkHealth().then(r => setStatus(r.status)).catch(() => setStatus('offline'));
    const t = setInterval(() => {
      checkHealth().then(r => setStatus(r.status)).catch(() => setStatus('offline'));
    }, 30000);
    return () => clearInterval(t);
  }, []);

  const handleSend = useCallback(async (query) => {
    setLoading(true);
    setError('');
    setActiveStep('retrieval');

    const userMsg = {
      role: 'user',
      content: query,
      files: files.map(f => ({ name: f.name, size: f.size })),
      time: now(),
    };
    setMessages(prev => [...prev, userMsg]);

    const aiMsgId = Date.now();
    setMessages(prev => [...prev, {
      id: aiMsgId,
      role: 'assistant',
      content: '',
      references: [],
      fallback: false,
      time: now(),
      loading: true,
    }]);

    try {
      if (files.length > 0) setActiveStep('ocr');

      const res = await sendMessage(query, subject, files, sessionId);

      setMessages(prev => prev.map(m =>
        m.id === aiMsgId ? {
          ...m,
          content: res.answer || '未获得回答',
          references: (res.references || []).map(r => ({
            citation: `[${r.file_name || '来源'}] ${r.text || ''}`,
            file_name: r.file_name,
            text: r.text,
            score: r.score,
          })),
          fallback: res.fallback_used || false,
          time: now(),
          loading: false,
        } : m
      ));

      setActiveStep('reasoning');
      setTimeout(() => setActiveStep('done'), 500);

    } catch (err) {
      setError(err.message);
      setActiveStep('');
      setMessages(prev => prev.map(m =>
        m.id === aiMsgId ? { ...m, content: `请求失败: ${err.message}`, loading: false, fallback: true } : m
      ));
    } finally {
      setLoading(false);
      setFiles([]);
      setShowFiles(false);
    }
  }, [files, subject, sessionId]);

  return (
    <div className="h-screen flex bg-[#0a0e17]">
      {/* 左侧栏 */}
      <Sidebar
        selectedSubject={subject}
        onSelectSubject={setSubject}
        onUploadClick={() => setShowFiles(!showFiles)}
        status={status}
      />

      {/* 主区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 顶部标题栏 */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-[#1e2d4a] bg-[#0a0e17]/80 backdrop-blur-xl shrink-0">
          <div className="flex items-center gap-3">
            <Zap size={18} className="text-cyan-400" />
            <h2 className="text-sm font-semibold text-white">
              {subject ? `${subject} · 投研分析` : '智能投研分析系统'}
            </h2>
            {subject && (
              <span className="px-2 py-0.5 rounded-md bg-purple-500/10 border border-purple-500/20 text-[10px] text-purple-400">
                {subject}
              </span>
            )}
          </div>

          <div className="hidden md:flex items-center gap-2">
            {[Cpu, Shield, Zap].map((Icon, i) => (
              <span key={i} className="text-[10px] text-slate-500 bg-white/5 px-2 py-1 rounded-md flex items-center gap-1">
                <Icon size={11} className={i === 0 ? 'text-cyan-400' : i === 1 ? 'text-purple-400' : 'text-amber-400'} />
                {['LangGraph', 'Milvus', 'Qwen-Max'][i]}
              </span>
            ))}
          </div>
        </header>

        {/* 聊天区域 */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
              <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-cyan-400/20 to-purple-500/20 flex items-center justify-center mb-6">
                <Zap size={36} className="text-cyan-400" />
              </div>
              <h2 className="text-xl font-bold text-white mb-2">智能投研分析系统</h2>
              <p className="text-sm text-slate-500 max-w-md mb-8">
                基于 Multi-Agent + RAG，上传上市公司年报即可获得深度投研分析。
                覆盖 10+ 行业，500+ 年报，万级财务指标索引。
              </p>

              <div className="grid grid-cols-2 gap-3 max-w-lg">
                {[
                  '对比宁德时代和比亚迪2024年营收增速',
                  '分析招商银行近三年ROE变化趋势',
                  '列出新能源行业平均毛利率Top5公司',
                  '贵州茅台2024年净利润增长驱动因素',
                ].map((q, i) => (
                  <button
                    key={i}
                    onClick={() => handleSend(q)}
                    className="text-left px-4 py-3 rounded-xl border border-[#1e2d4a] hover:border-cyan-500/30 hover:bg-cyan-500/[0.03] text-xs text-slate-400 hover:text-slate-200 transition-all"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatMessage key={msg.id || i} msg={msg} />
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-xs text-slate-500 mb-4 animate-fade-in">
              <Loader2 size={14} className="animate-spin text-cyan-400" />
              {activeStep === 'ocr' ? '正在解析文件...' : activeStep === 'retrieval' ? '正在检索知识...' : activeStep === 'reasoning' ? '正在生成回答...' : '处理中...'}
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Agent Pipeline */}
        {(loading || activeStep === 'done') && (
          <div className="px-6">
            <AgentPipeline activeStep={loading ? activeStep : 'done'} error={error} />
          </div>
        )}

        {/* File Upload */}
        {showFiles && (
          <FileUpload files={files} onFilesChange={setFiles} uploading={false} />
        )}

        {/* Input */}
        <ChatInput
          onSend={handleSend}
          disabled={loading}
          hasFiles={files.length > 0}
          onToggleFiles={() => setShowFiles(!showFiles)}
        />
      </div>
    </div>
  );
}
