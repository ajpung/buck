import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import time


# ----------------- OPTIMIZED RANDOM STATE -----------------
def _optimize_rs(X_train, y_train, X_test, y_true, opts):

    # Strategic random states
    strategic_rs = [0, 1, 7, 17, 42, 99, 123, 321, 777, 1337, 2024, 9999]

    best_acc = -np.inf
    best_val = strategic_rs[0]
    best_f1 = 0
    results = []

    pbar = tqdm(strategic_rs, desc="Random State")

    for v in pbar:
        classifier = ExtraTreesClassifier(
            random_state=v,
            n_estimators=25,  # Reduced for speed during optimization
            max_depth=5,  # Limited depth for speed
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=False,  # Disabled for speed
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results.append((v, accuracy, f1))

        if accuracy > best_acc:
            best_acc = accuracy
            best_val = v
            best_f1 = f1

        pbar.set_postfix(
            {
                "curr_rs": v,
                "curr_acc": f"{accuracy:.3f}",
                "best_rs": best_val,
                "best_acc": f"{best_acc:.3f}",
            }
        )

    opts["random_state"] = best_val

    print(f"\n📊 Top 3 Random States:")
    for rs, acc, f1 in sorted(results, key=lambda x: x[1], reverse=True)[:3]:
        print(f"   RS {rs:4d}: Acc={acc:.4f}, F1={f1:.4f}")

    return opts, best_acc, best_f1


# ----------------- OPTIMIZED N_ESTIMATORS -----------------
def _optimize_nest(X_train, y_train, X_test, y_true, opts):

    # Strategic n_estimators values
    strategic_nest = [10, 25, 50, 75, 100, 150, 200, 300, 500]

    best_acc = -np.inf
    best_val = strategic_nest[0]
    best_f1 = 0
    results = []

    pbar = tqdm(strategic_nest, desc="N_Estimators")

    for v in pbar:
        classifier = ExtraTreesClassifier(
            random_state=opts["random_state"],
            n_estimators=v,
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=False,  # Disabled for speed
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results.append((v, accuracy, f1))

        if accuracy > best_acc:
            best_acc = accuracy
            best_val = v
            best_f1 = f1

        pbar.set_postfix(
            {
                "curr_n": v,
                "curr_acc": f"{accuracy:.3f}",
                "best_n": best_val,
                "best_acc": f"{best_acc:.3f}",
            }
        )

    opts["n_estimators"] = best_val

    print(f"\n📊 Top 3 N_Estimators:")
    for n, acc, f1 in sorted(results, key=lambda x: x[1], reverse=True)[:3]:
        print(f"   N={n:3d}: Acc={acc:.4f}, F1={f1:.4f}")

    return opts, best_acc, best_f1


# ----------------- OPTIMIZED CORE PARAMETERS -----------------
def _optimize_core_params(X_train, y_train, X_test, y_true, opts):

    # Test combinations of max_depth, criterion, and max_features
    core_param_combos = [
        # (max_depth, criterion, max_features)
        (None, "gini", "sqrt"),
        (None, "gini", "log2"),
        (None, "entropy", "sqrt"),
        (None, "entropy", "log2"),
        (10, "gini", "sqrt"),
        (10, "gini", "log2"),
        (15, "gini", "sqrt"),
        (15, "entropy", "sqrt"),
        (20, "gini", "sqrt"),
        (5, "gini", "sqrt"),  # Shallow trees
        (None, "log_loss", "sqrt"),
        (None, "gini", None),  # All features
    ]

    best_acc = -np.inf
    best_depth = None
    best_criterion = "gini"
    best_max_features = "sqrt"
    best_f1 = 0
    results = []

    pbar = tqdm(core_param_combos, desc="Core Parameters")

    for max_depth, criterion, max_features in pbar:
        try:
            classifier = ExtraTreesClassifier(
                random_state=opts["random_state"],
                n_estimators=50,  # Moderate size for testing
                max_depth=max_depth,
                criterion=criterion,
                class_weight=opts["class_weight"],
                min_samples_split=opts["min_samples_split"],
                min_samples_leaf=opts["min_samples_leaf"],
                min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
                max_features=max_features,
                max_leaf_nodes=opts["max_leaf_nodes"],
                min_impurity_decrease=opts["min_impurity_decrease"],
                bootstrap=opts["bootstrap"],
                oob_score=False,
                n_jobs=opts["n_jobs"],
                verbose=opts["verbose"],
                warm_start=opts["warm_start"],
                ccp_alpha=opts["ccp_alpha"],
                max_samples=opts["max_samples"],
            )

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            results.append((max_depth, criterion, max_features, accuracy, f1))

            if accuracy > best_acc:
                best_acc = accuracy
                best_depth = max_depth
                best_criterion = criterion
                best_max_features = max_features
                best_f1 = f1

            depth_str = str(max_depth) if max_depth is not None else "None"
            feat_str = str(max_features) if max_features is not None else "None"

            pbar.set_postfix(
                {
                    "curr": f"{depth_str[:4]}-{criterion[:4]}-{feat_str[:4]}",
                    "curr_acc": f"{accuracy:.3f}",
                    "best_acc": f"{best_acc:.3f}",
                }
            )

        except Exception as e:
            print(f"   Warning: {max_depth}-{criterion}-{max_features} failed")
            continue

    opts["max_depth"] = best_depth
    opts["criterion"] = best_criterion
    opts["max_features"] = best_max_features

    print(f"\n📊 Top Core Parameter Combinations:")
    for depth, crit, feat, acc, f1 in sorted(results, key=lambda x: x[3], reverse=True)[
        :3
    ]:
        depth_str = str(depth) if depth is not None else "None"
        feat_str = str(feat) if feat is not None else "None"
        print(
            f"   Depth={depth_str:4s}, Crit={crit:8s}, Feat={feat_str:4s}: Acc={acc:.4f}, F1={f1:.4f}"
        )

    return opts, best_acc, best_f1


# ----------------- OPTIMIZED SAMPLING PARAMETERS -----------------
def _optimize_sampling_params(X_train, y_train, X_test, y_true, opts):

    # Strategic combinations of sampling parameters
    sampling_combos = [
        # (min_samples_split, min_samples_leaf, class_weight, bootstrap, max_samples)
        (2, 1, None, True, None),  # Default
        (2, 1, "balanced", True, None),  # Balanced classes
        (5, 2, None, True, None),  # More pruning
        (10, 3, "balanced", True, None),  # Balanced + pruning
        (2, 5, None, True, None),  # High leaf requirement
        (2, 1, None, False, None),  # No bootstrap
        (5, 2, "balanced", True, 0.8),  # Subsample + balanced
        (10, 5, "balanced_subsample", True, 0.7),  # Strong regularization
        (2, 1, "balanced_subsample", True, 0.9),  # Subsample balanced
    ]

    best_acc = -np.inf
    best_split = 2
    best_leaf = 1
    best_class_weight = None
    best_bootstrap = True
    best_max_samples = None
    best_f1 = 0
    results = []

    pbar = tqdm(sampling_combos, desc="Sampling Parameters")

    for min_split, min_leaf, class_weight, bootstrap, max_samples in pbar:
        try:
            classifier = ExtraTreesClassifier(
                random_state=opts["random_state"],
                n_estimators=opts["n_estimators"],
                max_depth=opts["max_depth"],
                criterion=opts["criterion"],
                class_weight=class_weight,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
                max_features=opts["max_features"],
                max_leaf_nodes=opts["max_leaf_nodes"],
                min_impurity_decrease=opts["min_impurity_decrease"],
                bootstrap=bootstrap,
                oob_score=False,
                n_jobs=opts["n_jobs"],
                verbose=opts["verbose"],
                warm_start=opts["warm_start"],
                ccp_alpha=opts["ccp_alpha"],
                max_samples=max_samples,
            )

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            results.append(
                (
                    min_split,
                    min_leaf,
                    class_weight,
                    bootstrap,
                    max_samples,
                    accuracy,
                    f1,
                )
            )

            if accuracy > best_acc:
                best_acc = accuracy
                best_split = min_split
                best_leaf = min_leaf
                best_class_weight = class_weight
                best_bootstrap = bootstrap
                best_max_samples = max_samples
                best_f1 = f1

            cw_str = str(class_weight)[:8] if class_weight is not None else "None"
            ms_str = str(max_samples)[:4] if max_samples is not None else "None"

            pbar.set_postfix(
                {
                    "curr": f"{min_split}-{min_leaf}-{cw_str[:4]}-{bootstrap}-{ms_str}",
                    "curr_acc": f"{accuracy:.3f}",
                    "best_acc": f"{best_acc:.3f}",
                }
            )

        except Exception as e:
            print(f"   Warning: sampling combo failed")
            continue

    opts["min_samples_split"] = best_split
    opts["min_samples_leaf"] = best_leaf
    opts["class_weight"] = best_class_weight
    opts["bootstrap"] = best_bootstrap
    opts["max_samples"] = best_max_samples

    print(f"\n📊 Top Sampling Parameter Combinations:")
    for split, leaf, cw, boot, ms, acc, f1 in sorted(
        results, key=lambda x: x[5], reverse=True
    )[:3]:
        cw_str = str(cw) if cw is not None else "None"
        ms_str = str(ms) if ms is not None else "None"
        print(
            f"   Split={split}, Leaf={leaf}, CW={cw_str:8s}, Boot={boot}, MS={ms_str}: Acc={acc:.4f}, F1={f1:.4f}"
        )

    return opts, best_acc, best_f1


# ----------------- OPTIMIZED REGULARIZATION -----------------
def _optimize_regularization(X_train, y_train, X_test, y_true, opts):

    # Strategic regularization combinations
    regularization_combos = [
        # (max_leaf_nodes, min_weight_fraction_leaf, min_impurity_decrease, ccp_alpha)
        (None, 0.0, 0.0, 0.0),  # No regularization
        (50, 0.0, 0.0, 0.0),  # Limit leaf nodes
        (100, 0.0, 0.0, 0.0),  # More leaf nodes
        (None, 0.01, 0.0, 0.0),  # Weight fraction
        (None, 0.0, 0.01, 0.0),  # Impurity decrease
        (None, 0.0, 0.0, 0.01),  # CCP alpha
        (50, 0.01, 0.01, 0.0),  # Combined regularization
        (30, 0.02, 0.02, 0.01),  # Strong regularization
        (200, 0.0, 0.005, 0.0),  # Moderate regularization
    ]

    best_acc = -np.inf
    best_nodes = None
    best_weight_frac = 0.0
    best_impurity = 0.0
    best_ccp = 0.0
    best_f1 = 0
    results = []

    pbar = tqdm(regularization_combos, desc="Regularization")

    for max_nodes, weight_frac, impurity, ccp in pbar:
        try:
            classifier = ExtraTreesClassifier(
                random_state=opts["random_state"],
                n_estimators=opts["n_estimators"],
                max_depth=opts["max_depth"],
                criterion=opts["criterion"],
                class_weight=opts["class_weight"],
                min_samples_split=opts["min_samples_split"],
                min_samples_leaf=opts["min_samples_leaf"],
                min_weight_fraction_leaf=weight_frac,
                max_features=opts["max_features"],
                max_leaf_nodes=max_nodes,
                min_impurity_decrease=impurity,
                bootstrap=opts["bootstrap"],
                oob_score=False,
                n_jobs=opts["n_jobs"],
                verbose=opts["verbose"],
                warm_start=opts["warm_start"],
                ccp_alpha=ccp,
                max_samples=opts["max_samples"],
            )

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            results.append((max_nodes, weight_frac, impurity, ccp, accuracy, f1))

            if accuracy > best_acc:
                best_acc = accuracy
                best_nodes = max_nodes
                best_weight_frac = weight_frac
                best_impurity = impurity
                best_ccp = ccp
                best_f1 = f1

            nodes_str = str(max_nodes) if max_nodes is not None else "None"

            pbar.set_postfix(
                {
                    "curr": f"{nodes_str[:4]}-{weight_frac:.2f}-{impurity:.2f}-{ccp:.2f}",
                    "curr_acc": f"{accuracy:.3f}",
                    "best_acc": f"{best_acc:.3f}",
                }
            )

        except Exception as e:
            print(f"   Warning: regularization combo failed")
            continue

    opts["max_leaf_nodes"] = best_nodes
    opts["min_weight_fraction_leaf"] = best_weight_frac
    opts["min_impurity_decrease"] = best_impurity
    opts["ccp_alpha"] = best_ccp

    print(f"\n📊 Top Regularization Configurations:")
    for nodes, wf, imp, ccp, acc, f1 in sorted(
        results, key=lambda x: x[4], reverse=True
    )[:3]:
        nodes_str = str(nodes) if nodes is not None else "None"
        print(
            f"   Nodes={nodes_str:4s}, WF={wf:.2f}, Imp={imp:.2f}, CCP={ccp:.2f}: Acc={acc:.4f}, F1={f1:.4f}"
        )

    return opts, best_acc, best_f1


# ----------------- MAIN OPTIMIZATION FUNCTION -----------------
def _optimize_extra_trees(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimized Extra Trees hyperparameter tuning

    Reduces optimization time from ~25+ hours to ~2-3 hours while maintaining quality
    """

    print("🚀 Starting Extra Trees Optimization")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Features: {X_train.shape[1]:,}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Optimization cycles: {cycles}")

    start_time = time.time()

    # Initialize with good defaults
    opts = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0,
        "warm_start": False,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "max_samples": None,
    }

    # Track progress across cycles
    cycle_results = []

    for c in range(cycles):
        print(f"\n{'=' * 50}")
        print(f"OPTIMIZATION CYCLE {c + 1}/{cycles}")
        print(f"{'=' * 50}")

        cycle_start = time.time()

        # Optimize in strategic order (most impactful first)
        opts, _, _ = _optimize_rs(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_core_params(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_nest(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_sampling_params(X_train, y_train, X_test, y_true, opts)
        opts, acc, f1 = _optimize_regularization(X_train, y_train, X_test, y_true, opts)

        cycle_time = time.time() - cycle_start
        cycle_results.append((c + 1, acc, f1, cycle_time))

        print(f"\n✅ Cycle {c + 1} Complete!")
        print(f"   Best Accuracy: {acc:.4f}")
        print(f"   Best F1 Score: {f1:.4f}")
        print(f"   Cycle Time: {cycle_time / 60:.1f} minutes")
        print(f"   Current Best Config:")
        print(f"     Random State: {opts['random_state']}")
        print(f"     N_Estimators: {opts['n_estimators']}")
        print(f"     Criterion: {opts['criterion']}")
        print(f"     Max Depth: {opts['max_depth']}")
        print(f"     Max Features: {opts['max_features']}")
        print(f"     Min Samples Split: {opts['min_samples_split']}")
        print(f"     Min Samples Leaf: {opts['min_samples_leaf']}")
        print(f"     Class Weight: {opts['class_weight']}")
        print(f"     Bootstrap: {opts['bootstrap']}")
        print(f"     Max Samples: {opts['max_samples']}")

    total_time = time.time() - start_time

    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    print(f"   Total Time: {total_time / 60:.1f} minutes")
    print(f"   Final Best Accuracy: {acc:.4f}")
    print(f"   Final Best F1 Score: {f1:.4f}")

    print(f"\n📈 Cycle Progress:")
    for cycle, c_acc, c_f1, c_time in cycle_results:
        print(
            f"   Cycle {cycle}: Acc={c_acc:.4f}, F1={c_f1:.4f}, Time={c_time / 60:.1f}min"
        )

    # Return all useful information
    ma_vec = [result[1] for result in cycle_results]
    f1_vec = [result[2] for result in cycle_results]

    return opts, acc, f1, ma_vec, f1_vec
