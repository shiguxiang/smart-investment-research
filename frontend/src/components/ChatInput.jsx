import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, X, Sparkles } from 'lucide-react';

export default function ChatInput({ onSend, disabled, hasFiles, onToggleFiles }) {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  const handleSend = () => {
    if (!input.trim() || disabled) return;
    onSend(input.trim());
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-4 border-t border-[#1e2d4a] glass">
      <div className="flex items-end gap-3 bg-[#0a0e17] rounded-2xl border border-[#1e2d4a] px-4 py-2.5 focus-within:border-cyan-500/40 focus-within:shadow-[0_0_15px_rgba(0,229,255,0.1)] transition-all">
        {/* 附件按钮 */}
        <button
          onClick={onToggleFiles}
          className={`p-1.5 rounded-lg transition-all ${hasFiles ? 'bg-purple-500/20 text-purple-400 shadow-[0_0_8px_rgba(167,139,250,0.3)]' : 'text-slate-600 hover:text-slate-400'}`}
        >
          <Paperclip size={18} />
        </button>

        {/* 输入框 */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入你的问题... (Shift+Enter 换行)"
          rows={1}
          disabled={disabled}
          className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 resize-none outline-none max-h-[120px] py-1"
        />

        {/* 发送按钮 */}
        <button
          onClick={handleSend}
          disabled={!input.trim() || disabled}
          className={`p-2 rounded-xl transition-all duration-300 ${
            input.trim() && !disabled
              ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50 hover:scale-105'
              : 'bg-white/5 text-slate-600'
          }`}
        >
          {disabled ? (
            <Sparkles size={18} className="animate-pulse" />
          ) : (
            <Send size={18} />
          )}
        </button>
      </div>

      {/* 提示 */}
      <div className="flex items-center justify-between mt-2 px-1">
        <p className="text-[10px] text-slate-600">
          500+ 年报 · 10+ 行业 · 万级指标 · 投研智能分析
        </p>
        <p className="text-[10px] text-slate-600">
          {input.length > 0 ? `${input.length} 字符` : 'Enter 发送'}
        </p>
      </div>
    </div>
  );
}
