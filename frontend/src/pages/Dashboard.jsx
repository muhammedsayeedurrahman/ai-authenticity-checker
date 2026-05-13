import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, Image, Film, Mic, Layers, TrendingUp, BarChart3, Clock } from 'lucide-react';
import { Link } from 'react-router-dom';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import useForensicStore from '../store/useForensicStore';

export default function Dashboard() {
  const { history, historyTotal, fetchHistory } = useForensicStore();

  useEffect(() => {
    fetchHistory(10);
  }, [fetchHistory]);

  const cards = [
    {
      to: "/image",
      title: "Image Forensics",
      desc: "Detect manipulated pixels and synthetic faces",
      icon: <Image size={32} className="text-accent" />,
      bg: "from-accent/5 to-transparent",
    },
    {
      to: "/video",
      title: "Video Forensics",
      desc: "Frame-by-frame temporal consistency alignment",
      icon: <Film size={32} className="text-indigo-400" />,
      bg: "from-indigo-400/5 to-transparent",
    },
    {
      to: "/audio",
      title: "Audio Forensics",
      desc: "Spectrogram analysis for synthetic voice cloning",
      icon: <Mic size={32} className="text-accent-success" />,
      bg: "from-accent-success/5 to-transparent",
    },
    {
      to: "/multimodal",
      title: "Multimodal Fusion",
      desc: "Compound confidence matrix using all available factors",
      icon: <Layers size={32} className="text-accent-warning" />,
      bg: "from-accent-warning/5 to-transparent",
    },
  ];

  // Compute verdict distribution for pie chart
  const verdictCounts = history.reduce((acc, item) => {
    const v = item.verdict?.toUpperCase() || '';
    if (v.includes('HIGH') || v.includes('CRITICAL')) acc.high += 1;
    else if (v.includes('MEDIUM')) acc.medium += 1;
    else acc.low += 1;
    return acc;
  }, { low: 0, medium: 0, high: 0 });

  const pieData = [
    { name: 'Low Risk', value: verdictCounts.low },
    { name: 'Medium Risk', value: verdictCounts.medium },
    { name: 'High Risk', value: verdictCounts.high },
  ].filter(d => d.value > 0);

  const PIE_COLORS = ['#22C55E', '#EAB308', '#EF4444'];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-start min-h-[85vh] text-center space-y-12 pt-8">
      <div className="space-y-6">
        <div className="w-24 h-24 mx-auto rounded-3xl bg-gradient-to-br from-accent to-indigo-400 flex items-center justify-center shadow-[0_0_50px_rgba(99,102,241,0.3)]">
          <ShieldAlert className="text-white" size={48} />
        </div>
        <h2 className="text-5xl font-black tracking-tighter">
          Intelligence <span className="bg-clip-text text-transparent bg-gradient-to-r from-accent to-indigo-400">Command Center</span>
        </h2>
        <p className="text-text-secondary max-w-xl mx-auto text-lg">
          Select an analysis vector below to initialize the neural detection pipeline.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-6xl">
        {cards.map((card) => (
          <Link
            key={card.to}
            to={card.to}
            className="group relative glass-card p-6 border-border-subtle overflow-hidden transition-all duration-500 hover:scale-105 hover:-translate-y-2 hover:border-accent/30"
          >
            <div className={`absolute inset-0 bg-gradient-to-b ${card.bg} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
            <div className="relative z-10 flex flex-col items-center text-center space-y-4">
              <div className="p-4 rounded-xl bg-background border border-[rgba(255,255,255,0.05)] group-hover:scale-110 transition-transform duration-500">
                {card.icon}
              </div>
              <div>
                <h3 className="font-bold text-white mb-2 tracking-wide">{card.title}</h3>
                <p className="text-xs text-text-muted">{card.desc}</p>
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Analytics Section */}
      {historyTotal > 0 && (
        <div className="w-full max-w-6xl grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="glass-card p-6 flex flex-col items-center">
            <TrendingUp size={24} className="text-accent mb-3" />
            <p className="text-3xl font-black text-text-primary">{historyTotal}</p>
            <p className="text-xs text-text-muted uppercase tracking-wider mt-1">Total Analyses</p>
          </div>

          <div className="glass-card p-6 flex flex-col items-center">
            <BarChart3 size={24} className="text-accent mb-3" />
            <p className="text-sm font-bold text-text-secondary mb-2">Risk Distribution</p>
            {pieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={120}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={30} outerRadius={50} dataKey="value" stroke="none">
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#18181B', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', fontSize: '12px' }} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-text-muted text-xs">No data yet</p>
            )}
          </div>

          <div className="glass-card p-6">
            <div className="flex items-center gap-2 mb-4">
              <Clock size={16} className="text-accent" />
              <p className="text-sm font-bold text-text-secondary">Recent Activity</p>
            </div>
            <div className="space-y-2 max-h-[140px] overflow-y-auto">
              {history.slice(0, 5).map((item) => (
                <div key={item.id} className="flex justify-between items-center text-xs py-1.5 border-b border-border-subtle last:border-0">
                  <span className="text-text-primary capitalize">{item.media_type}</span>
                  <span className={`font-bold ${
                    item.risk_score > 0.7 ? 'text-accent-danger' : item.risk_score > 0.4 ? 'text-accent-warning' : 'text-accent-success'
                  }`}>
                    {(item.risk_score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}
