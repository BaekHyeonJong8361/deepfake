"""
ensemble_sweep_all.py
=====================
all_models_ood_v2.json 에 저장된 scores 로
v8~v36 전 모델 2-way / 3-way 앙상블 + 임계값 탐색.
GPU 불필요 — 순수 CPU numpy.
"""
import json
import numpy as np
from itertools import combinations
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

SRC = '/home/t26106/deepfake/saved_models/all_models_ood_v2.json'
OUT = '/home/t26106/deepfake/saved_models/ensemble_sweep_all.json'

def calc(labels, scores):
    lb = np.array(labels); sc = np.array(scores)
    auc = roc_auc_score(lb, sc)
    fpr, tpr, thrs = roc_curve(lb, sc)
    idx = int(np.argmax(tpr - fpr))
    thr = float(thrs[idx])
    acc = float(accuracy_score(lb, (sc >= thr).astype(int)))
    return float(auc), thr, acc

def sweep_2way(sc1, sc2, labels, name1, name2, weights=None):
    if weights is None:
        weights = np.arange(0, 1.01, 0.05)
    lb = np.array(labels)
    best = {'auc': 0}
    results = []
    for w in weights:
        sc = w * sc1 + (1-w) * sc2
        auc, thr, acc = calc(lb, sc)
        results.append({'w1': round(float(w),2), 'w2': round(1-float(w),2), 'auc': auc, 'thr': thr, 'acc': acc})
        if auc > best['auc']:
            best = {f'w_{name1}': round(float(w),2), f'w_{name2}': round(1-float(w),2),
                    'auc': auc, 'thr': thr, 'acc': acc}
    results.sort(key=lambda x: x['auc'], reverse=True)
    return best, results[:5]

def main():
    d = json.load(open(SRC))
    results = d['results']

    # 상위 AUC 모델만 필터 (0.70 이상)
    models = {n: r for n, r in results.items() if r['auc'] >= 0.70}
    labels = list(results['v24']['labels'])  # 공통 labels

    print('=' * 78)
    print(f'Ensemble sweep: {len(models)} 모델 (AUC≥0.70)')
    print(f'n={len(labels)} (SNS OOD v2)')
    print('=' * 78)

    # 단일 모델 순위
    print('\n[단일 모델 순위]')
    singles = sorted([(n, r['auc']) for n, r in models.items()], key=lambda x: x[1], reverse=True)
    for n, auc in singles:
        print(f'  {n:<15s} AUC={auc:.4f}')

    # 2-way 앙상블 전체 조합
    print('\n[2-way 앙상블 — 상위 20개]')
    model_names = [n for n, _ in singles]
    two_way_results = []

    for n1, n2 in combinations(model_names, 2):
        sc1 = np.array(results[n1]['scores'])
        sc2 = np.array(results[n2]['scores'])
        best, _ = sweep_2way(sc1, sc2, labels, n1, n2)
        two_way_results.append({
            'models': [n1, n2], **best
        })

    two_way_results.sort(key=lambda x: x['auc'], reverse=True)
    for r in two_way_results[:20]:
        m1, m2 = r['models']
        w1 = r[f'w_{m1}']; w2 = r[f'w_{m2}']
        print(f'  {m1}({w1}) + {m2}({w2})  AUC={r["auc"]:.4f}  thr={r["thr"]:.4f}  acc={r["acc"]:.1%}')

    # 3-way 앙상블 — 상위 5개 모델만
    top5 = [n for n, _ in singles[:5]]
    print(f'\n[3-way 앙상블 — 상위 5개 모델 조합: {top5}]')
    three_way_results = []

    for n1, n2, n3 in combinations(top5, 3):
        sc1 = np.array(results[n1]['scores'])
        sc2 = np.array(results[n2]['scores'])
        sc3 = np.array(results[n3]['scores'])
        lb  = np.array(labels)
        best_3 = {'auc': 0}
        for w1 in np.arange(0, 1.01, 0.1):
            for w2 in np.arange(0, 1.01-w1, 0.1):
                w3 = round(1 - w1 - w2, 1)
                if w3 < 0: continue
                sc  = w1*sc1 + w2*sc2 + w3*sc3
                auc, thr, acc = calc(lb, sc)
                if auc > best_3['auc']:
                    best_3 = {f'w_{n1}': round(w1,1), f'w_{n2}': round(w2,1), f'w_{n3}': round(w3,1),
                              'auc': auc, 'thr': thr, 'acc': acc, 'models': [n1,n2,n3]}
        three_way_results.append(best_3)

    three_way_results.sort(key=lambda x: x['auc'], reverse=True)
    for r in three_way_results[:10]:
        m1, m2, m3 = r['models']
        print(f'  {m1}({r[f"w_{m1}"]}) + {m2}({r[f"w_{m2}"]}) + {m3}({r[f"w_{m3}"]})  AUC={r["auc"]:.4f}  thr={r["thr"]:.4f}  acc={r["acc"]:.1%}')

    # 최적 결과 요약
    best_2 = two_way_results[0]
    best_3 = three_way_results[0] if three_way_results else None

    print('\n' + '=' * 78)
    print('[최종 요약]')
    print('=' * 78)
    print(f'  단일 최고:  v24  AUC={results["v24"]["auc"]:.4f}')
    m1, m2 = best_2['models']
    print(f'  2-way 최고: {m1}+{m2}  AUC={best_2["auc"]:.4f}  thr={best_2["thr"]:.4f}')
    if best_3:
        m1, m2, m3 = best_3['models']
        print(f'  3-way 최고: {m1}+{m2}+{m3}  AUC={best_3["auc"]:.4f}  thr={best_3["thr"]:.4f}')

    out = {
        'single': {n: {'auc': results[n]['auc'], 'thr': results[n]['thr']} for n in models},
        'best_2way': best_2,
        'top20_2way': two_way_results[:20],
        'best_3way': best_3,
        'top10_3way': three_way_results[:10],
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n저장: {OUT}')

if __name__ == '__main__':
    main()
