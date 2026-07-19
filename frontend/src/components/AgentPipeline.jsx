import { ScanEye, Search, Brain, CheckCircle2, Loader2, AlertTriangle } from 'lucide-react';

const stages = [
  { key: 'ocr', label: '文档解析', icon: ScanEye, color: '#a78bfa', desc: '版面分析+表格提取+跨页拼接' },
  { key: 'retrieval', label: '财务分析', icon: Search, color: '#00e5ff', desc: '三路混合召回+重排' },
  { key: 'reasoning', label: '综合研判', icon: Brain, color: '#34d399', desc: 'AI 生成投研分析报告' },
];

export default function AgentPipeline({ activeStep, error }) {
  const currentIdx = stages.findIndex(s => s.key === activeStep);

  return (
    <div className="glass rounded-xl p-4 mb-3 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Agent 工作流</p>
        {error && (
          <span className="flex items-center gap-1 text-xs text-rose-400">
            <AlertTriangle size={12} /> {error}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        {stages.map((stage, i) => {
          const Icon = stage.icon;
          const isActive = i === currentIdx && !error;
          const isDone = i < currentIdx;
          const isFailed = error && i === currentIdx;

          return (
            <div key={stage.key} className="flex-1 flex items-center">
              {/* Node */}
              <div className="flex flex-col items-center flex-1">
                <div
                  className={`w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-500 ${
                    isDone
                      ? 'bg-green-500/20 border border-green-500/30'
                      : isActive
                      ? 'bg-opacity-20 border-2 pulse-glow'
                      : isFailed
                      ? 'bg-rose-500/20 border border-rose-500/30'
                      : 'bg-white/5 border border-[#1e2d4a]'
                  }`}
                  style={isActive ? { borderColor: stage.color, backgroundColor: `${stage.color}15`, boxShadow: `0 0 15px ${stage.color}30` } : {}}
                >
                  {isDone ? (
                    <CheckCircle2 size={20} className="text-green-400" />
                  ) : isActive ? (
                    <Loader2 size={18} className="animate-spin" style={{ color: stage.color }} />
                  ) : isFailed ? (
                    <AlertTriangle size={18} className="text-rose-400" />
                  ) : (
                    <Icon size={18} style={{ color: stage.color, opacity: 0.5 }} />
                  )}
                </div>
                <span
                  className="text-[10px] mt-1.5 font-medium transition-colors"
                  style={{ color: isActive ? stage.color : isDone ? '#34d399' : '#475569' }}
                >
                  {stage.label}
                </span>
                <span className="text-[9px] text-slate-600 mt-0.5">{stage.desc}</span>
              </div>

              {/* Connector */}
              {i < stages.length - 1 && (
                <div className="w-8 h-px mx-0.5 mb-4" style={{ background: `linear-gradient(90deg, ${stages[i].color}40, ${stages[i+1].color}40)` }} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
