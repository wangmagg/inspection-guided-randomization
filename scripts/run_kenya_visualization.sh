#!/bin/bash

declare -a fig_type_arr=("hist_z") # "scatter_2dscore_disagg" "hist_score" "scatter_2dscore_disagg" "hist_same_z_validity" "scatter_error_vs_score") 
declare -a param_subdir_arr=("ki-mu-pos_ko-da-pos_sd-sm_bal")
declare -a po_mdl_subdir_arr=("kenya-hierarchical-nbr-sum")
declare -a net_mdl_subdir_arr=(
    "net-nested-2lvl-sb_psi-0.20_pdiso-0.02_pdido-0.01_intxn-euclidean-dist-power-decay-gamma-0.5" \
    "net-nested-2lvl-sb_psi-0.20_pdiso-0.02_pdido-0.01_intxn-euclidean-dist-power-decay-gamma-0.25" \
    "net-nested-2lvl-sb_psi-0.20_pdiso-0.02_pdido-0.01_intxn-euclidean-dist-power-decay-gamma-0.75"
    )

for fig_type in "${fig_type_arr[@]}"
do
    for param_subdir in "${param_subdir_arr[@]}"
    do
        for po_mdl_subdir in "${po_mdl_subdir_arr[@]}"
        do
            for net_mdl_subdir in "${net_mdl_subdir_arr[@]}"
            do
                # With conditional addition of mirrors
                # Frac Expo (lin comb)
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs \
                        rand-restricted_lin-comb_max-mahalanobis-0.25_frac-exposed-0.75_cluster-school \
                        rand-restricted_lin-comb_max-mahalanobis-0.50_frac-exposed-0.50_cluster-school \
                        rand-restricted_lin-comb_max-mahalanobis-0.75_frac-exposed-0.25_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # Min Euclid (lin comb)
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs \
                        rand-restricted_lin-comb_max-mahalanobis-0.25_min-pairwise-euclidean-dist-0.75_cluster-school \
                        rand-restricted_lin-comb_max-mahalanobis-0.50_min-pairwise-euclidean-dist-0.50_cluster-school \
                        rand-restricted_lin-comb_max-mahalanobis-0.75_min-pairwise-euclidean-dist-0.25_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # Frac Expo 
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs rand-restricted_frac-exposed_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # Min Euclid
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs rand-restricted_min-pairwise-euclidean-dist_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # Mahalanobis
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs rand-restricted_max-mahalanobis_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # With all mirrors added
                # Frac Expo (lin comb)
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs \
                        rand-restricted_lin-comb_max-mahalanobis-0.25_frac-exposed-0.75-all-mirr_cluster-school \
                        rand-restricted_lin-comb_max-mahalanobis-0.50_frac-exposed-0.50-all-mirr_cluster-school \
                        rand-restricted_lin-comb_max-mahalanobis-0.75_frac-exposed-0.25-all-mirr_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # Min Euclid (lin comb)
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs \
                        rand-restricted_lin-comb_max-mahalanobis-0.25_min-pairwise-euclidean-dist-0.75-all-mirr_cluster-school \
                        rand-restricted_lin-comb_max-mahalanobis-0.50_min-pairwise-euclidean-dist-0.50-all-mirr_cluster-school \
                        rand-restricted_lin-comb_max-mahalanobis-0.75_min-pairwise-euclidean-dist-0.25-all-mirr_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # Frac Expo 
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs rand-restricted_frac-exposed-all-mirr_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # Min Euclid
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs rand-restricted_min-pairwise-euclidean-dist-all-mirr_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]} 

                # Mahalanobis
                python3 -m src.visualization.make_plots \
                    --param-subdir $param_subdir \
                    --po-mdl-subdir $po_mdl_subdir \
                    --fig-type ${fig_type_arr[@]} \
                    --rand-mdl-subdirs rand-restricted_max-mahalanobis-all-mirr_cluster-school \
                    --net-mdl-subdir ${net_mdl_subdir[@]}
            done
        done
    done
done

# python3 -m src.visualization.make_plots \
#     --exp-subdir kenya \
#     --fig-types school_locs covariate_distr_across_sets adj_v_dist pairwise_dists


