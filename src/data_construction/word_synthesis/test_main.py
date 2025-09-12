# test_main.py
from pathlib import Path
from src.data_construction.word_synthesis.word_synthesis import WordSynthesis
from src.data_construction.word_synthesis.cosine_similarity_pseudoword import PseudoWordCosineRunner
from src.data_construction.word_synthesis.candidate_filter import CandidateFilterRunner
from tqdm import tqdm

def main():
    INDEX     = "v4_rpj_llama_s4"
    MODEL_ID  = "meta-llama/Llama-3.2-1B-Instruct"
    CACHE_DIR = Path("./models/linguistic_models")
    P2G_DIR   = CACHE_DIR / "english_us_arpa"

    DEBUG   = False
    N_WORDS = 10 if DEBUG else 200

    synth = WordSynthesis(
        index=INDEX,
        model_id=MODEL_ID,
        cache_dir=str(CACHE_DIR),
        model_dir=str(P2G_DIR),
        seed=3457,
        debug=DEBUG,
    )

    # 1) 生成伪词
    # pseudo = synth.generation_pseudowords(n=N_WORDS, nbest=5, min_len=3, max_len=12)
    pseudo = ['eroc', 'ima', 'uck', 'orzon', 'irkohith', 'edzs', 'audai', 'uhit', 'ittititze', 'unziz', 'nmip', 'itaer', 'imped', 'ermputonec', 'urrmaam', 'undondillms', 'earctotsure', 'iburr', 'outhe', 'istarliteng', 'angid', 'aud', 'unasan', 'ished', 'unuth', 'eekyed', 'oosoolzensk', 'utsyte', 'erderk', 'uhin', 'ektick', 'awrare', 'etoirrlden', 'osts', 'osdermirth', 'ikoeon', 'ucour', 'airturped', 'uddaiznick', 'auction', 'ildaife', 'imollalls', 'utulch', 'emebe', 'uch', 'iseen', 'easopes', 'een', 'easyck', 'iquescern', 'adzz', 'everrbelk', 'imbe', 'onde', 'acambe', 'anzah', 'udog', 'empta', 'icum', 'onsb', 'euxnsud', 'uhudv', 'ush', 'onell', 'epuh', 'eatr', 'ombrend', 'aroyhut', 'illinvilt', 'ucte', 'onsubidne', 'oaktull', 'akkeefss', 'untilox', 'usang', 'utmert', 'emanin', 'utleazes', 'owiermuntl', 'uonnette', 'ain', 'isk', 'oag', 'easthullique', 'aftutoang', 'eamdealt', 'eined', 'ulede', 'ocnuh', 'ermemplonm', 'enittake', 'eintun', 'uluzhid', 'ounegas', 'untsud', 'unud', 'oogifetick', 'urrai', 'ownsud', 'ashaste', 'ihe', 'entuuf', 'ojung', 'ifseh', 'itruenslalse', 'uriersk', 'awaup', 'uoh', 'erosne', 'irrelldre', 'eyetnas', 'unwe', 'udjok', 'aylant', 'uvl', 'achrudjoux', 'imoul', 'urrah', 'aeck', 'imscind', 'ulurjuze', 'innagh', 'epstirks', 'uzai', 'uirrav', 'usact', 'etosuz', 'egrettsch', 'eaderr', 'uhnd', 'elpholp', 'eth', 'ainn', 'ezerr', 'aldor', 'ubsticheep', 'eyelzon', 'eyetzov', 'usheece', 'unimen', 'ifr', 'eard', 'aneuved', 'indnaish', 'uswabskins', 'eastsung', 'uxirred', 'eethee', 'anernull', 'onuz', 'emphuh', 'ignan', 'ufoes', 'nuz', 'usurce', 'ovahn', 'uhonne', 'ool', 'itaunned', 'oolald', 'erbim', 'irzah', 'epefem', 'ikaes', 'urse', 'urtidone', 'undees', 'uzzahducah', 'erpone', 'erndselled', 'ariquhiu', 'iksd', 'iobe', 'ekite', 'ass', 'uxcurllal', 'ukinung', 'einse', 'edjacooks', 'itre', 'assimball', 'udge', 'ernsteaux', 'uzz', 'anaque', 'ezoim', 'otzaung', 'oei', 'osh', 'ilepp', 'ulaw', 'onitnute', 'oaorchauffed', 'innearse', 'inteh', 'onzanit', 'earlungeux', 'immurn', 'isurrell', 'iquilisk']
    print(f"[INFO] generated {len(pseudo)} pseudo-words (debug={DEBUG})")
    print("[SAMPLE 10] ->", pseudo)

    

    """
    # 2) 抓已有语料 top-k 词（第一阶段 CSV）
    csv_path = synth.fetch_existing_top_words(topk=10000)  # debug=True 时内部用 Top-100
    print(f"[INFO] word_count CSV -> {csv_path}")

    # 3) union（debug=True ⇒ 只看100个；base_topk=10 对应你已存在的 top_words_10.csv）
    out_csv, out_csv_all = synth.build_union_with_wordfreq(
        topk=11000, word_freq_n=50000, base_topk=10, debug=True
    )
    print("[INFO] union out:", out_csv)
    print("[INFO] union all:", out_csv_all)
    """
    
    out_csv_all = "results/word_synthesis/v4_rpj_llama_s4/union_all_words_wf50000.csv"
    
    """
    # 4) 生成 embedding + 聚合矩阵（embedding/last 两路）
    # !!! 这里要用 union 的“all”CSV，而不是 top_words
    matrix_dirs = synth.build_embeddings_and_matrix(csv_in=out_csv_all)
    print("[INFO] matrix dirs:", matrix_dirs)
    
    

    # 5) 确保真实词矩阵可用（已构建则直接复用；缺失则按 union_all 顺序聚合）
    synth.ensure_realword_matrix(union_csv=out_csv_all, force_rebuild=False, cleanup_chunks=False)

    # 6) 伪词 vs 真实词 余弦TopK（一次得到 embedding + last 两路；debug下K取小一点）
    # sim_csv = synth.compare_pseudowords(pseudo_words=pseudo, k=50)
    # print("[INFO] similarities CSV ->", sim_csv)
    
    
    runner = PseudoWordCosineRunner(index="v4_rpj_llama_s4",
                                model_id="meta-llama/Llama-3.2-1B-Instruct",
                                pooling="mean", debug=DEBUG)


    # 1) 仅保存 “整向量”
    for p_word in tqdm(pseudo):
        runner.save_full_vectors(p_word)

        # 2) 仅做 t-SNE 可视化
        runner.save_tsne(p_word, source="embedding", n_real=800, top_k=200)
        runner.save_tsne(p_word, source="last", n_real=800, top_k=200)

        # 3) 一把梭：向量 + 可视化都生成
        # runner.save_vectors_and_tsne("eroc", n_real=10, top_k=20)
    """
    outs = CandidateFilterRunner(
        index=INDEX,
        model_id=MODEL_ID,
        debug=DEBUG,
        topk_neighbors=200,
        qps=6, retry=5,
        rank_dirname="sort_candidate",
        run_real_control=True, real_sample_n=200, real_seed=42, real_dirname="real_sort",
    ).run()
    print(outs)


if __name__ == "__main__":
    main()
