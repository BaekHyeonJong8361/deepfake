import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
// import type { VideoAnalysisType } from '../../lib/api';
import { useVideoStore } from '../../stores/videoStore';
import styles from './ScanInput.module.css';


  const DESCRIPTIONS = {
    RYZE: "고전 딥페이크(FF++, DFDC) 특화 모델입니다.",
    LEE_SIN: "찰나의 조작도 잡아내는 공격적 탐지 모드입니다.",
    SHEN: "오늘 실험 1위! 가장 정확한 밸런스형 모델입니다.",
    RAMMUS: "확실한 조작만 잡아내는 보수적 방어 모드입니다.",
    T2V: "생성형 AI(Text-to-Video) 전용 탐지 엔진입니다."
  };

export default function ScanInput() {
  const [url, setUrl] = useState('');
  const [category, setCategory] = useState<'DEEPFAKE' | 'T2V'>('DEEPFAKE');
  const [subModel, setSubModel] = useState<'RYZE' | 'LEE_SIN' | 'SHEN' | 'RAMMUS'>('SHEN');
  const [agreed, setAgreed] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { submitVideoUrl, uploadVideoFile, isSubmitting, error } = useVideoStore();

  const handleScan = async () => {
    if (!agreed || !url.trim() || isSubmitting) return;
    try {
      await submitVideoUrl(url.trim(), category === 'T2V' ? 'T2V' : subModel);
      navigate('/scan/analysis');
    } catch {
      // Managed by store
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !agreed || isSubmitting) return;
    try {
      await uploadVideoFile(file, category === 'T2V' ? 'T2V' : subModel);
      navigate('/scan/analysis');
    } catch {
      // Managed by store
    }
  };

  return (
    <div className={styles.inputGroup}>
      <div className={styles.inputRow}>
        <input
          type="text"
          placeholder="https://www.example.com/video.mp4"
          aria-label="Video URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <input
          ref={fileInputRef}
          className={styles.fileInput}
          type="file"
          accept="video/*"
          onChange={handleFileChange}
        />
        <button
          className={styles.iconBtn}
          aria-label="Upload File"
          type="button"
          disabled={!agreed || isSubmitting}
          onClick={() => fileInputRef.current?.click()}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </button>
      </div>
      <div className={styles.actionRow}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%' }}>
          {/* Category Toggle */}
          <div className={styles.typeGroup} style={{ marginBottom: 0 }}>
            {['DEEPFAKE', 'T2V'].map((cat) => (
              <button
                key={cat}
                type="button"
                className={styles.typeButton + (category === cat ? " " + styles.typeButtonActive : "")}
                onClick={() => setCategory(cat as any)}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Sub-Model selection for Deepfake */}
          {category === 'DEEPFAKE' && (
            <div style={{ display: 'flex', gap: '8px', padding: '10px', background: 'rgba(0,0,0,0.03)', borderRadius: '12px', border: '1px solid rgba(0,0,0,0.05)' }}>
              {['RYZE', 'LEE_SIN', 'SHEN', 'RAMMUS'].map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setSubModel(m as any)}
                  style={{
                    flex: 1,
                    padding: '6px 4px',
                    fontSize: '0.65rem',
                    borderRadius: '8px',
                    border: '1px solid ' + (subModel === m ? 'var(--accent-blue)' : 'transparent'),
                    background: subModel === m ? 'var(--accent-blue)' : 'transparent',
                    color: subModel === m ? '#fff' : 'var(--text-muted)',
                    cursor: 'pointer',
                    fontWeight: subModel === m ? 700 : 400,
                    transition: 'all 0.2s'
                  }}
                >
                  {m}
                </button>
              ))}
            </div>
          )}
        </div>
        <div style={{ marginTop: '8px', padding: '8px 12px', background: 'rgba(0, 3, 255, 0.05)', borderRadius: '8px', borderLeft: '3px solid var(--accent-blue)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-main)', margin: 0, fontWeight: 500 }}>
            💡 {category === 'T2V' ? DESCRIPTIONS.T2V : DESCRIPTIONS[subModel]}
          </p>
        </div>
        <label className={styles.consent}>
          <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
          I agree to terms of analysis
        </label>
        <button className={styles.btnPrimary} onClick={handleScan} disabled={!agreed || !url.trim() || isSubmitting}>
          {isSubmitting ? 'REQUESTING' : 'SCAN NOW'} <span className={styles.beta}>BETA</span>
        </button>
      </div>
      {error && <p className={styles.errorText}>{error}</p>}
    </div>
  );
}
