import { BookOpen, Upload, BarChart3, Activity, TrendingUp } from 'lucide-react';

const industries = ['银行业', '新能源', '半导体', '医药生物', '消费电子', '房地产', '汽车制造', '食品饮料', '计算机', '化工'];

export default function Sidebar({ selectedSubject, onSelectSubject, onUploadClick, status }) {
  return (
    <aside className="w-64 h-full glass flex flex-col shrink-0 border-r border-[#1e2d4a]">
      {/* Logo */}
      <div className="p-5 border-b border-[#1e2d4a]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-purple-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
            <TrendingUp size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white leading-tight">智能投研分析</h1>
            <p className="text-[10px] text-slate-500">Multi-Agent + RAG</p>
          </div>
        </div>
      </div>

      {/* 状态指示器 */}
      <div className="px-4 py-3 border-b border-[#1e2d4a] flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${status === 'connected' ? 'bg-green-400 shadow-[0_0_8px_#34d399]' : status === 'degraded' ? 'bg-amber-400 shadow-[0_0_8px_#fbbf24]' : 'bg-red-400 shadow-[0_0_8px_#f43f5e]'}`} />
        <span className="text-xs text-slate-400">
          {status === 'connected' ? '系统就绪' : status === 'degraded' ? '降级运行' : '离线'}
        </span>
      </div>

      {/* 行业列表 */}
      <div className="flex-1 overflow-y-auto p-3">
        <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2 px-2">行业分类</p>
        {industries.map(s => (
          <button
            key={s}
            onClick={() => onSelectSubject(s === selectedSubject ? '' : s)}
            className={`w-full text-left px-3 py-2 rounded-lg mb-1 text-sm transition-all duration-200 ${
              s === selectedSubject
                ? 'bg-gradient-to-r from-cyan-500/10 to-purple-500/10 text-cyan-300 border border-cyan-500/20 shadow-[0_0_10px_rgba(0,229,255,0.05)]'
                : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      {/* 底部操作 */}
      <div className="p-3 border-t border-[#1e2d4a] space-y-1">
        <button
          onClick={onUploadClick}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 hover:bg-white/5 hover:text-white transition-all"
        >
          <Upload size={16} className="text-purple-400" />
          上传年报
        </button>
        <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-400 hover:bg-white/5 hover:text-white transition-all">
          <BarChart3 size={16} className="text-cyan-400" />
          评估报告
        </button>
        <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-400 hover:bg-white/5 hover:text-white transition-all">
          <Activity size={16} className="text-amber-400" />
          系统监控
        </button>
      </div>
    </aside>
  );
}
