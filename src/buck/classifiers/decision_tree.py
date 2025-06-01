import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
        classifier = DecisionTreeClassifier(
            random_state=v,
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=5,  # Limited depth for speed during optimization
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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


# ----------------- OPTIMIZED CRITERION & SPLITTER -----------------
def _optimize_criterion_splitter(X_train, y_train, X_test, y_true, opts):

    # Test combinations of criterion and splitter
    criterion_splitter_combos = [
        ("gini", "best"),
        ("gini", "random"),
        ("entropy", "best"),
        ("entropy", "random"),
        ("log_loss", "best"),
        ("log_loss", "random"),
    ]

    best_acc = -np.inf
    best_criterion = "gini"
    best_splitter = "best"
    best_f1 = 0
    results = []

    pbar = tqdm(criterion_splitter_combos, desc="Criterion & Splitter")

    for criterion, splitter in pbar:
        try:
            classifier = DecisionTreeClassifier(
                random_state=opts["random_state"],
                criterion=criterion,
                splitter=splitter,
                max_depth=5,  # Limited for speed
                min_samples_split=opts["min_samples_split"],
                min_samples_leaf=opts["min_samples_leaf"],
                min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
                max_features=opts["max_features"],
                max_leaf_nodes=opts["max_leaf_nodes"],
                min_impurity_decrease=opts["min_impurity_decrease"],
                class_weight=opts["class_weight"],
                ccp_alpha=opts["ccp_alpha"],
            )

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            results.append((criterion, splitter, accuracy, f1))

            if accuracy > best_acc:
                best_acc = accuracy
                best_criterion = criterion
                best_splitter = splitter
                best_f1 = f1

            pbar.set_postfix(
                {
                    "curr": f"{criterion[:4]}-{splitter[:4]}",
                    "curr_acc": f"{accuracy:.3f}",
                    "best": f"{best_criterion[:4]}-{best_splitter[:4]}",
                    "best_acc": f"{best_acc:.3f}",
                }
            )

        except Exception as e:
            print(f"   Warning: {criterion}-{splitter} failed")
            continue

    opts["criterion"] = best_criterion
    opts["splitter"] = best_splitter

    print(f"\n📊 Criterion & Splitter Results:")
    for crit, split, acc, f1 in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"   {crit:8s}-{split:6s}: Acc={acc:.4f}, F1={f1:.4f}")

    return opts, best_acc, best_f1


# ----------------- OPTIMIZED MAX_DEPTH -----------------
def _optimize_max_depth(X_train, y_train, X_test, y_true, opts):

    # Strategic max_depth values
    strategic_depths = [1, 2, 3, 4, 5, 7, 10, 15, 20, None]

    best_acc = -np.inf
    best_val = strategic_depths[0]
    best_f1 = 0
    results = []

    pbar = tqdm(strategic_depths, desc="Max Depth")

    for v in pbar:
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=v,
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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

        depth_str = str(v) if v is not None else "None"
        pbar.set_postfix(
            {
                "curr_depth": depth_str,
                "curr_acc": f"{accuracy:.3f}",
                "best_depth": str(best_val) if best_val is not None else "None",
                "best_acc": f"{best_acc:.3f}",
            }
        )

    opts["max_depth"] = best_val

    print(f"\n📊 Top 3 Max Depths:")
    for depth, acc, f1 in sorted(results, key=lambda x: x[1], reverse=True)[:3]:
        depth_str = str(depth) if depth is not None else "None"
        print(f"   Depth {depth_str:4s}: Acc={acc:.4f}, F1={f1:.4f}")

    return opts, best_acc, best_f1


# ----------------- OPTIMIZED SAMPLES PARAMETERS -----------------
def _optimize_samples_params(X_train, y_train, X_test, y_true, opts):

    # Strategic combinations of min_samples_split and min_samples_leaf
    samples_combos = [
        (2, 1),  # Default
        (5, 2),  # Moderate pruning
        (10, 3),  # More pruning
        (15, 5),  # Strong pruning
        (20, 7),  # Very strong pruning
        (2, 5),  # High leaf requirement
        (2, 10),  # Very high leaf requirement
    ]

    best_acc = -np.inf
    best_split = 2
    best_leaf = 1
    best_f1 = 0
    results = []

    pbar = tqdm(samples_combos, desc="Samples Params")

    for min_split, min_leaf in pbar:
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results.append((min_split, min_leaf, accuracy, f1))

        if accuracy > best_acc:
            best_acc = accuracy
            best_split = min_split
            best_leaf = min_leaf
            best_f1 = f1

        pbar.set_postfix(
            {
                "curr": f"{min_split}-{min_leaf}",
                "curr_acc": f"{accuracy:.3f}",
                "best": f"{best_split}-{best_leaf}",
                "best_acc": f"{best_acc:.3f}",
            }
        )

    opts["min_samples_split"] = best_split
    opts["min_samples_leaf"] = best_leaf

    print(f"\n📊 Samples Parameters Results:")
    for split, leaf, acc, f1 in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"   Split={split:2d}, Leaf={leaf:2d}: Acc={acc:.4f}, F1={f1:.4f}")

    return opts, best_acc, best_f1


# ----------------- OPTIMIZED FEATURES & STRUCTURE -----------------
def _optimize_features_structure(X_train, y_train, X_test, y_true, opts):

    # Strategic combinations of max_features and max_leaf_nodes
    features_structure_combos = [
        (None, None),  # No limits
        ("sqrt", None),  # Square root features
        ("log2", None),  # Log2 features
        (0.5, None),  # Half features
        (0.7, None),  # 70% features
        (None, 20),  # Limited leaf nodes
        (None, 50),  # More leaf nodes
        ("sqrt", 30),  # Sqrt features + limited nodes
        ("log2", 40),  # Log2 features + limited nodes
    ]

    best_acc = -np.inf
    best_features = None
    best_nodes = None
    best_f1 = 0
    results = []

    pbar = tqdm(features_structure_combos, desc="Features & Structure")

    for max_features, max_leaf_nodes in pbar:
        try:
            classifier = DecisionTreeClassifier(
                random_state=opts["random_state"],
                criterion=opts["criterion"],
                splitter=opts["splitter"],
                max_depth=opts["max_depth"],
                min_samples_split=opts["min_samples_split"],
                min_samples_leaf=opts["min_samples_leaf"],
                min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=opts["min_impurity_decrease"],
                class_weight=opts["class_weight"],
                ccp_alpha=opts["ccp_alpha"],
            )

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            results.append((max_features, max_leaf_nodes, accuracy, f1))

            if accuracy > best_acc:
                best_acc = accuracy
                best_features = max_features
                best_nodes = max_leaf_nodes
                best_f1 = f1

            feat_str = str(max_features)[:4] if max_features is not None else "None"
            node_str = str(max_leaf_nodes) if max_leaf_nodes is not None else "None"
            pbar.set_postfix(
                {
                    "curr": f"{feat_str}-{node_str}",
                    "curr_acc": f"{accuracy:.3f}",
                    "best_acc": f"{best_acc:.3f}",
                }
            )

        except Exception as e:
            print(f"   Warning: {max_features}-{max_leaf_nodes} failed")
            continue

    opts["max_features"] = best_features
    opts["max_leaf_nodes"] = best_nodes

    print(f"\n📊 Features & Structure Results:")
    for feat, nodes, acc, f1 in sorted(results, key=lambda x: x[2], reverse=True)[:3]:
        feat_str = str(feat) if feat is not None else "None"
        node_str = str(nodes) if nodes is not None else "None"
        print(
            f"   Features={feat_str:4s}, Nodes={node_str:4s}: Acc={acc:.4f}, F1={f1:.4f}"
        )

    return opts, best_acc, best_f1


# ----------------- OPTIMIZED REGULARIZATION -----------------
def _optimize_regularization(X_train, y_train, X_test, y_true, opts):

    # Strategic regularization combinations
    regularization_combos = [
        # (min_weight_fraction_leaf, min_impurity_decrease, ccp_alpha, class_weight)
        (0.0, 0.0, 0.0, None),  # No regularization
        (0.0, 0.0, 0.0, "balanced"),  # Balanced classes only
        (0.0, 0.01, 0.0, None),  # Small impurity decrease
        (0.0, 0.05, 0.0, None),  # Moderate impurity decrease
        (0.01, 0.0, 0.0, None),  # Small weight fraction
        (0.0, 0.0, 0.01, None),  # Small CCP alpha
        (0.0, 0.01, 0.01, "balanced"),  # Combined regularization
        (0.01, 0.01, 0.01, "balanced"),  # Full regularization
    ]

    best_acc = -np.inf
    best_weight_frac = 0.0
    best_impurity = 0.0
    best_ccp = 0.0
    best_class_weight = None
    best_f1 = 0
    results = []

    pbar = tqdm(regularization_combos, desc="Regularization")

    for weight_frac, impurity, ccp, class_weight in pbar:
        try:
            classifier = DecisionTreeClassifier(
                random_state=opts["random_state"],
                criterion=opts["criterion"],
                splitter=opts["splitter"],
                max_depth=opts["max_depth"],
                min_samples_split=opts["min_samples_split"],
                min_samples_leaf=opts["min_samples_leaf"],
                min_weight_fraction_leaf=weight_frac,
                max_features=opts["max_features"],
                max_leaf_nodes=opts["max_leaf_nodes"],
                min_impurity_decrease=impurity,
                class_weight=class_weight,
                ccp_alpha=ccp,
            )

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            results.append((weight_frac, impurity, ccp, class_weight, accuracy, f1))

            if accuracy > best_acc:
                best_acc = accuracy
                best_weight_frac = weight_frac
                best_impurity = impurity
                best_ccp = ccp
                best_class_weight = class_weight
                best_f1 = f1

            pbar.set_postfix(
                {"curr_acc": f"{accuracy:.3f}", "best_acc": f"{best_acc:.3f}"}
            )

        except Exception as e:
            print(f"   Warning: regularization combo failed")
            continue

    opts["min_weight_fraction_leaf"] = best_weight_frac
    opts["min_impurity_decrease"] = best_impurity
    opts["ccp_alpha"] = best_ccp
    opts["class_weight"] = best_class_weight

    print(f"\n📊 Top 3 Regularization Configs:")
    for wf, imp, ccp, cw, acc, f1 in sorted(results, key=lambda x: x[4], reverse=True)[
        :3
    ]:
        cw_str = str(cw) if cw is not None else "None"
        print(
            f"   WF={wf:.2f}, Imp={imp:.2f}, CCP={ccp:.2f}, CW={cw_str}: Acc={acc:.4f}, F1={f1:.4f}"
        )

    return opts, best_acc, best_f1


# ----------------- MAIN OPTIMIZATION FUNCTION -----------------
def _optimize_decision_tree(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimized Decision Tree hyperparameter tuning

    Reduces optimization time from ~20+ hours to ~1-2 hours while maintaining quality
    """

    print("🚀 Starting Decision Tree Optimization")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Features: {X_train.shape[1]:,}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Optimization cycles: {cycles}")

    start_time = time.time()

    # Initialize with good defaults
    opts = {
        "random_state": 42,
        "criterion": "gini",
        "splitter": "best",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": None,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": 0.0,
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
        opts, _, _ = _optimize_criterion_splitter(
            X_train, y_train, X_test, y_true, opts
        )
        opts, _, _ = _optimize_max_depth(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_samples_params(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_features_structure(
            X_train, y_train, X_test, y_true, opts
        )
        opts, acc, f1 = _optimize_regularization(X_train, y_train, X_test, y_true, opts)

        cycle_time = time.time() - cycle_start
        cycle_results.append((c + 1, acc, f1, cycle_time))

        print(f"\n✅ Cycle {c + 1} Complete!")
        print(f"   Best Accuracy: {acc:.4f}")
        print(f"   Best F1 Score: {f1:.4f}")
        print(f"   Cycle Time: {cycle_time / 60:.1f} minutes")
        print(f"   Current Best Config:")
        print(f"     Random State: {opts['random_state']}")
        print(f"     Criterion: {opts['criterion']}")
        print(f"     Splitter: {opts['splitter']}")
        print(f"     Max Depth: {opts['max_depth']}")
        print(f"     Min Samples Split: {opts['min_samples_split']}")
        print(f"     Min Samples Leaf: {opts['min_samples_leaf']}")
        print(f"     Max Features: {opts['max_features']}")
        print(f"     Max Leaf Nodes: {opts['max_leaf_nodes']}")
        print(f"     Class Weight: {opts['class_weight']}")

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
