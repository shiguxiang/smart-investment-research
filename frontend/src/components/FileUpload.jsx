import { useState, useRef } from 'react';
import { Upload, X, FileText, Image, FileSpreadsheet, Loader2, CheckCircle2 } from 'lucide-react';

const iconMap = {
  'application/pdf': FileText,
  'image/': Image,
  'ppt': FileSpreadsheet,
};

function getIcon(type) {
  for (const [k, v] of Object.entries(iconMap)) {
    if (type.includes(k)) return v;
  }
  return FileText;
}

export default function FileUpload({ files, onFilesChange, uploading }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = Array.from(e.dataTransfer.files);
    onFilesChange([...files, ...dropped]);
  };

  const handleRemove = (i) => {
    onFilesChange(files.filter((_, idx) => idx !== i));
  };

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files);
    onFilesChange([...files, ...selected]);
    e.target.value = '';
  };

  return (
    <div className="px-4 py-3">
      {/* 已选文件 */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {files.map((f, i) => {
            const Icon = getIcon(f.type);
            return (
              <div key={i} className="flex items-center gap-2 bg-white/5 border border-[#1e2d4a] rounded-lg px-3 py-1.5 text-xs animate-fade-in">
                <Icon size={12} className="text-purple-400" />
                <span className="text-slate-300 max-w-[120px] truncate">{f.name}</span>
                <button onClick={() => handleRemove(i)} className="text-slate-600 hover:text-rose-400 transition-colors">
                  <X size={12} />
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* 拖拽区域 */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-300 ${
          dragOver
            ? 'border-cyan-400 bg-cyan-500/5 shadow-[0_0_30px_rgba(0,229,255,0.1)]'
            : 'border-[#1e2d4a] hover:border-slate-600 hover:bg-white/[0.02]'
        }`}
      >
        {uploading ? (
          <div className="flex flex-col items-center gap-2">
            <Loader2 size={24} className="animate-spin text-cyan-400" />
            <span className="text-xs text-slate-400">处理中...</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500/10 to-cyan-500/10 flex items-center justify-center">
              <Upload size={22} className="text-purple-400" />
            </div>
            <p className="text-xs text-slate-400">
              <span className="text-purple-400 font-medium">点击上传</span> 或拖拽文件到此处
            </p>
            <p className="text-[10px] text-slate-600">支持年报 PDF · 扫描件 PNG · JPG</p>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.ppt,.pptx,.png,.jpg,.jpeg,.bmp,.tiff"
        onChange={handleFileSelect}
        className="hidden"
      />
    </div>
  );
}
