#!/bin/bash

vignette_name=$1

if [ "$vignette_name" == "mult-arm" ]
then
    # Vignette 1 (Mult-arm)
    python3 -m src.visualization.make_manu_plots \
        --exp-subdir mult-arm \
        --rand-mdl-subdirs rand-restricted_max-mahalanobis rand-restricted_sum-max-abs-smd \
        --fig-type hist_score hist_z hist_same_z_validity covariate_balance scatter_error_vs_score  
fi

if [ "$vignette_name" == "composition" ]
then
    # Vignette 2 (Composition)
    python3 -m src.visualization.make_manu_plots \
        --exp-subdir composition \
        --rand-mdl-subdirs \
            rand-group-formation-restricted_max-mahalanobis \
            rand-group-formation-restricted_sum-max-abs-smd \
            rand-group-formation-restricted-genetic_max-mahalanobis \
            rand-group-formation-restricted-genetic_sum-max-abs-smd \
        --fig-type covariate_balance 
    python3 -m src.visualization.make_manu_plots \
        --exp-subdir composition \
        --rand-mdl-subdirs \
            rand-group-formation-restricted_max-mahalanobis \
            rand-group-formation-restricted_sum-max-abs-smd \
        --fig-type hist_score hist_same_z_validity 
fi

if [ "$vignette_name" == "kenya" ]
then
    # Vignette 3 (Kenya)
    declare -a fig_type_arr=("scatter_2dscore_disagg") #("hist_score" "scatter_2dscore_disagg" "hist_same_z_validity" "scatter_error_vs_score") 
    declare -a param_subdir_arr=("ki-mu-pos_ko-da-pos_sd-sm_bal")
    declare -a po_mdl_subdir_arr=("kenya-hierarchical-nbr-sum")
    declare -a net_mdl_subdir_arr=(
        "net-nested-2lvl-sb_psi-0.20_pdiso-0.02_pdido-0.01_intxn-euclidean-dist-power-decay-gamma-0.5" \
        "net-nested-2lvl-sb_psi-0.20_pdiso-0.02_pdido-0.01_intxn-euclidean-dist-power-decay-gamma-0.25" \
        "net-nested-2lvl-sb_psi-0.20_pdiso-0.02_pdido-0.01_intxn-euclidean-dist-power-decay-gamma-0.75"
        )
    
    for param_subdir in "${param_subdir_arr[@]}"
    do
        for po_mdl_subdir in "${po_mdl_subdir_arr[@]}"
        do
            python3 -m src.visualization.make_manu_plots \
                --param-subdir $param_subdir \
                --po-mdl-subdir $po_mdl_subdir \
                --fig-type hist_2dscore_disagg \
                --net-mdl-subdir ${net_mdl_subdir_arr[@]} \
                --rand-mdl-subdirs rand-restricted_lin-comb_max-mahalanobis-0.50_frac-exposed-0.50_cluster-school
        done
    done

    for fig_type in "${fig_type_arr[@]}"
    do
        for param_subdir in "${param_subdir_arr[@]}"
        do
            for po_mdl_subdir in "${po_mdl_subdir_arr[@]}"
            do
                for net_mdl_subdir in "${net_mdl_subdir_arr[@]}"
                do

                    # Frac Expo (lin comb)
                    python3 -m src.visualization.make_manu_plots \
                        --param-subdir $param_subdir \
                        --po-mdl-subdir $po_mdl_subdir \
                        --fig-type ${fig_type_arr[@]} \
                        --rand-mdl-subdirs \
                            rand-restricted_lin-comb_max-mahalanobis-0.25_frac-exposed-0.75_cluster-school \
                            rand-restricted_lin-comb_max-mahalanobis-0.50_frac-exposed-0.50_cluster-school \
                            rand-restricted_lin-comb_max-mahalanobis-0.75_frac-exposed-0.25_cluster-school \
                        --net-mdl-subdir ${net_mdl_subdir[@]} \
                        --plt-suffix "max-mahalanobis_frac-exposed"

                    # Min Euclid (lin comb)
                    python3 -m src.visualization.make_manu_plots \
                        --param-subdir $param_subdir \
                        --po-mdl-subdir $po_mdl_subdir \
                        --fig-type ${fig_type_arr[@]} \
                        --rand-mdl-subdirs \
                            rand-restricted_lin-comb_max-mahalanobis-0.25_min-pairwise-euclidean-dist-0.75_cluster-school \
                            rand-restricted_lin-comb_max-mahalanobis-0.50_min-pairwise-euclidean-dist-0.50_cluster-school \
                            rand-restricted_lin-comb_max-mahalanobis-0.75_min-pairwise-euclidean-dist-0.25_cluster-school \
                        --net-mdl-subdir ${net_mdl_subdir[@]} \
                        --plt-suffix "max-mahalanobs_min-pairwise-euclidean-dist"

                    # Frac Expo 
                    python3 -m src.visualization.make_manu_plots \
                        --param-subdir $param_subdir \
                        --po-mdl-subdir $po_mdl_subdir \
                        --fig-type ${fig_type_arr[@]} \
                        --rand-mdl-subdirs rand-restricted_frac-exposed_cluster-school \
                        --net-mdl-subdir ${net_mdl_subdir[@]} \
                        --plt-suffix "frac-exposed"

                    # Min Euclid
                    python3 -m src.visualization.make_manu_plots \
                        --param-subdir $param_subdir \
                        --po-mdl-subdir $po_mdl_subdir \
                        --fig-type ${fig_type_arr[@]} \
                        --rand-mdl-subdirs rand-restricted_min-pairwise-euclidean-dist_cluster-school \
                        --net-mdl-subdir ${net_mdl_subdir[@]} \
                        --plt-suffix "min-pairwise-euclidean-dist"

                    # Mahalanobis
                    python3 -m src.visualization.make_manu_plots \
                        --param-subdir $param_subdir \
                        --po-mdl-subdir $po_mdl_subdir \
                        --fig-type ${fig_type_arr[@]} \
                        --rand-mdl-subdirs rand-restricted_max-mahalanobis_cluster-school \
                        --net-mdl-subdir ${net_mdl_subdir[@]} \
                        --plt-suffix "max-mahalanobis"
                done
            done
        done
    done
fi