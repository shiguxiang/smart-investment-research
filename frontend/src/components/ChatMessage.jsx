import { User, Bot, FileText, ExternalLink, ChevronDown, ChevronUp, Clock, AlertTriangle } from 'lucide-react';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ChatMessage({ msg }) {
  const [showRefs, setShowRefs] = useState(false);
  const isUser = msg.role === 'user';

  return (
    <div className={`flex gap-3 mb-4 animate-slide-up ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${
        isUser
          ? 'bg-gradient-to-br from-cyan-400 to-blue-500 shadow-lg shadow-cyan-500/20'
          : 'bg-gradient-to-br from-purple-500 to-pink-500 shadow-lg shadow-purple-500/20'
      }`}>
        {isUser ? <User size={16} className="text-white" /> : <Bot size={16} className="text-white" />}
      </div>

      {/* Content */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'items-end' : ''}`}>
        {/* Timestamp */}
        <div className={`flex items-center gap-2 mb-1 ${isUser ? 'justify-end' : ''}`}>
          <span className="text-[10px] text-slate-600">{isUser ? '分析师' : 'AI 投研助手'}</span>
          <Clock size={10} className="text-slate-600" />
          <span className="text-[10px] text-slate-600">{msg.time}</span>
        </div>

        {/* Message bubble */}
        <div className={`rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/20 rounded-tr-sm'
            : 'glass rounded-tl-sm'
        }`}>
          {isUser ? (
            <p className="text-sm text-slate-200 whitespace-pre-wrap">{msg.content}</p>
          ) : (
            <div className="markdown-body text-sm text-slate-300">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            </div>
          )}

          {/* Files indicator */}
          {msg.files && msg.files.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-[#1e2d4a]">
              {msg.files.map((f, i) => (
                <span key={i} className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-cyan-500/10 border border-cyan-500/20 text-[10px] text-cyan-400">
                  <FileText size={10} /> {f.name}
                </span>
              ))}
            </div>
          )}

          {/* Fallback warning */}
          {msg.fallback && (
            <div className="flex items-center gap-1.5 mt-2 pt-2 border-t border-amber-500/20 text-[11px] text-amber-400">
              <AlertTriangle size={12} /> 使用了降级策略，结果可能不够精确
            </div>
          )}
        </div>

        {/* References */}
        {!isUser && msg.references && msg.references.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowRefs(!showRefs)}
              className="inline-flex items-center gap-1 text-[11px] text-slate-500 hover:text-cyan-400 transition-colors"
            >
              <ExternalLink size={11} />
              {msg.references.length} 条引用来源
              {showRefs ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>

            {showRefs && (
              <div className="mt-2 space-y-1.5 animate-fade-in">
                {msg.references.map((ref, i) => (
                  <div key={i} className="text-[11px] text-slate-400 bg-black/20 rounded-lg px-3 py-2 border border-[#1e2d4a]">
                    <span className="text-cyan-400 font-medium">[{i + 1}]</span>{' '}
                    {ref.citation || ref.text || `${ref.file_name}: ${ref.text?.substring(0, 80)}...`}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
