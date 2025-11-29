# 研磨速度预测分析报告

**生成时间**: 2025-11-29 17:17:11

---

## 1. 执行摘要

本报告由AI Agent自动生成，基于提供的数据进行研磨速度预测建模分析。

### 关键发现

- **最佳模型**: RandomForest
- **模型性能**: R² = 0.8862
- **数据规模**: 793 条记录，128 个特征

---

## 2. 数据分析

### 2.1 数据概览

- **数据维度**: 793 行 × 128 列
- **特征列表**: WAFER_ID, f_duration, f_USAGE_OF_BACKING_FILM, f_USAGE_OF_DRESSER, f_USAGE_OF_POLISHING_TABLE, f_USAGE_OF_DRESSER_TABLE, f_USAGE_OF_MEMBRANE, f_USAGE_OF_PRESSURIZED_SHEET, f_mean_PRESSURIZED_CHAMBER_PRESSURE, f_mean_MAIN_OUTER_AIR_BAG_PRESSURE, f_mean_CENTER_AIR_BAG_PRESSURE, f_mean_RETAINER_RING_PRESSURE, f_mean_RIPPLE_AIR_BAG_PRESSURE, f_mean_EDGE_AIR_BAG_PRESSURE, f_mean_DRESSING_WATER_STATUS, f_mean_SLURRY_FLOW_LINE_A, f_mean_SLURRY_FLOW_LINE_B, f_mean_SLURRY_FLOW_LINE_C, f_mean_WAFER_ROTATION, f_mean_STAGE_ROTATION, f_mean_HEAD_ROTATION, f_median_PRESSURIZED_CHAMBER_PRESSURE, f_median_MAIN_OUTER_AIR_BAG_PRESSURE, f_median_CENTER_AIR_BAG_PRESSURE, f_median_RETAINER_RING_PRESSURE, f_median_RIPPLE_AIR_BAG_PRESSURE, f_median_EDGE_AIR_BAG_PRESSURE, f_median_SLURRY_FLOW_LINE_A, f_median_SLURRY_FLOW_LINE_B, f_median_SLURRY_FLOW_LINE_C, f_median_WAFER_ROTATION, f_median_STAGE_ROTATION, f_median_HEAD_ROTATION, f_std_PRESSURIZED_CHAMBER_PRESSURE, f_std_MAIN_OUTER_AIR_BAG_PRESSURE, f_std_CENTER_AIR_BAG_PRESSURE, f_std_RETAINER_RING_PRESSURE, f_std_RIPPLE_AIR_BAG_PRESSURE, f_std_EDGE_AIR_BAG_PRESSURE, f_std_DRESSING_WATER_STATUS, f_std_SLURRY_FLOW_LINE_A, f_std_SLURRY_FLOW_LINE_B, f_std_SLURRY_FLOW_LINE_C, f_std_WAFER_ROTATION, f_std_STAGE_ROTATION, f_std_HEAD_ROTATION, f_auc_PRESSURIZED_CHAMBER_PRESSURE, f_auc_MAIN_OUTER_AIR_BAG_PRESSURE, f_auc_CENTER_AIR_BAG_PRESSURE, f_auc_RETAINER_RING_PRESSURE, f_auc_RIPPLE_AIR_BAG_PRESSURE, f_auc_EDGE_AIR_BAG_PRESSURE, f_auc_SLURRY_FLOW_LINE_A, f_auc_SLURRY_FLOW_LINE_B, f_auc_SLURRY_FLOW_LINE_C, f_auc_WAFER_ROTATION, f_auc_STAGE_ROTATION, f_auc_HEAD_ROTATION, f_is_first_run, f_last_mrr, s_duration, s_mean_PRESSURIZED_CHAMBER_PRESSURE, s_mean_MAIN_OUTER_AIR_BAG_PRESSURE, s_mean_CENTER_AIR_BAG_PRESSURE, s_mean_RETAINER_RING_PRESSURE, s_mean_RIPPLE_AIR_BAG_PRESSURE, s_mean_EDGE_AIR_BAG_PRESSURE, s_mean_DRESSING_WATER_STATUS, s_mean_SLURRY_FLOW_LINE_A, s_mean_SLURRY_FLOW_LINE_B, s_mean_SLURRY_FLOW_LINE_C, s_mean_WAFER_ROTATION, s_mean_STAGE_ROTATION, s_mean_HEAD_ROTATION, s_median_PRESSURIZED_CHAMBER_PRESSURE, s_median_MAIN_OUTER_AIR_BAG_PRESSURE, s_median_CENTER_AIR_BAG_PRESSURE, s_median_RETAINER_RING_PRESSURE, s_median_RIPPLE_AIR_BAG_PRESSURE, s_median_EDGE_AIR_BAG_PRESSURE, s_median_SLURRY_FLOW_LINE_A, s_median_SLURRY_FLOW_LINE_B, s_median_SLURRY_FLOW_LINE_C, s_median_WAFER_ROTATION, s_median_STAGE_ROTATION, s_median_HEAD_ROTATION, s_std_PRESSURIZED_CHAMBER_PRESSURE, s_std_MAIN_OUTER_AIR_BAG_PRESSURE, s_std_CENTER_AIR_BAG_PRESSURE, s_std_RETAINER_RING_PRESSURE, s_std_RIPPLE_AIR_BAG_PRESSURE, s_std_EDGE_AIR_BAG_PRESSURE, s_std_DRESSING_WATER_STATUS, s_std_SLURRY_FLOW_LINE_A, s_std_SLURRY_FLOW_LINE_B, s_std_SLURRY_FLOW_LINE_C, s_std_WAFER_ROTATION, s_std_STAGE_ROTATION, s_std_HEAD_ROTATION, s_auc_PRESSURIZED_CHAMBER_PRESSURE, s_auc_MAIN_OUTER_AIR_BAG_PRESSURE, s_auc_CENTER_AIR_BAG_PRESSURE, s_auc_RETAINER_RING_PRESSURE, s_auc_RIPPLE_AIR_BAG_PRESSURE, s_auc_EDGE_AIR_BAG_PRESSURE, s_auc_SLURRY_FLOW_LINE_A, s_auc_SLURRY_FLOW_LINE_B, s_auc_SLURRY_FLOW_LINE_C, s_auc_WAFER_ROTATION, s_auc_STAGE_ROTATION, s_auc_HEAD_ROTATION, knn_0, knn_1, knn_2, knn_3, knn_4, knn_5, knn_6, knn_7, knn_8, knn_9, knn_10, len, wt_max, mode_knn_0, mode_knn_1, mode_knn_2, AVG_REMOVAL_RATE
- **内存占用**: 0.77 MB

### 2.2 数据质量


**异常值** (基于IQR方法):
- f_duration: 14 个异常值
- f_mean_PRESSURIZED_CHAMBER_PRESSURE: 21 个异常值
- f_mean_MAIN_OUTER_AIR_BAG_PRESSURE: 30 个异常值
- f_mean_CENTER_AIR_BAG_PRESSURE: 78 个异常值
- f_mean_RETAINER_RING_PRESSURE: 5 个异常值
- f_mean_RIPPLE_AIR_BAG_PRESSURE: 31 个异常值
- f_mean_EDGE_AIR_BAG_PRESSURE: 24 个异常值
- f_mean_SLURRY_FLOW_LINE_A: 121 个异常值
- f_mean_SLURRY_FLOW_LINE_B: 86 个异常值
- f_mean_SLURRY_FLOW_LINE_C: 10 个异常值
- f_mean_WAFER_ROTATION: 83 个异常值
- f_mean_STAGE_ROTATION: 125 个异常值
- f_mean_HEAD_ROTATION: 60 个异常值
- f_median_PRESSURIZED_CHAMBER_PRESSURE: 21 个异常值
- f_median_MAIN_OUTER_AIR_BAG_PRESSURE: 151 个异常值
- f_median_CENTER_AIR_BAG_PRESSURE: 59 个异常值
- f_median_RETAINER_RING_PRESSURE: 26 个异常值
- f_median_RIPPLE_AIR_BAG_PRESSURE: 31 个异常值
- f_median_EDGE_AIR_BAG_PRESSURE: 21 个异常值
- f_median_SLURRY_FLOW_LINE_C: 11 个异常值
- f_median_WAFER_ROTATION: 82 个异常值
- f_median_STAGE_ROTATION: 30 个异常值
- f_median_HEAD_ROTATION: 1 个异常值
- f_std_PRESSURIZED_CHAMBER_PRESSURE: 107 个异常值
- f_std_MAIN_OUTER_AIR_BAG_PRESSURE: 95 个异常值
- f_std_CENTER_AIR_BAG_PRESSURE: 107 个异常值
- f_std_RETAINER_RING_PRESSURE: 25 个异常值
- f_std_RIPPLE_AIR_BAG_PRESSURE: 99 个异常值
- f_std_EDGE_AIR_BAG_PRESSURE: 89 个异常值
- f_std_SLURRY_FLOW_LINE_A: 10 个异常值
- f_std_SLURRY_FLOW_LINE_C: 69 个异常值
- f_std_WAFER_ROTATION: 91 个异常值
- f_std_STAGE_ROTATION: 80 个异常值
- f_std_HEAD_ROTATION: 67 个异常值
- f_auc_PRESSURIZED_CHAMBER_PRESSURE: 20 个异常值
- f_auc_MAIN_OUTER_AIR_BAG_PRESSURE: 3 个异常值
- f_auc_CENTER_AIR_BAG_PRESSURE: 14 个异常值
- f_auc_RETAINER_RING_PRESSURE: 17 个异常值
- f_auc_RIPPLE_AIR_BAG_PRESSURE: 2 个异常值
- f_auc_EDGE_AIR_BAG_PRESSURE: 22 个异常值
- f_auc_SLURRY_FLOW_LINE_A: 1 个异常值
- f_auc_SLURRY_FLOW_LINE_B: 19 个异常值
- f_auc_SLURRY_FLOW_LINE_C: 3 个异常值
- f_auc_WAFER_ROTATION: 82 个异常值
- f_auc_STAGE_ROTATION: 60 个异常值
- f_auc_HEAD_ROTATION: 29 个异常值
- f_last_mrr: 20 个异常值
- s_duration: 31 个异常值
- s_mean_WAFER_ROTATION: 79 个异常值
- s_mean_STAGE_ROTATION: 81 个异常值
- s_mean_HEAD_ROTATION: 30 个异常值
- s_median_WAFER_ROTATION: 97 个异常值
- s_median_STAGE_ROTATION: 12 个异常值
- s_std_WAFER_ROTATION: 79 个异常值
- s_std_STAGE_ROTATION: 81 个异常值
- s_std_HEAD_ROTATION: 30 个异常值
- s_auc_PRESSURIZED_CHAMBER_PRESSURE: 2 个异常值
- s_auc_MAIN_OUTER_AIR_BAG_PRESSURE: 3 个异常值
- s_auc_CENTER_AIR_BAG_PRESSURE: 3 个异常值
- s_auc_RETAINER_RING_PRESSURE: 2 个异常值
- s_auc_RIPPLE_AIR_BAG_PRESSURE: 3 个异常值
- s_auc_EDGE_AIR_BAG_PRESSURE: 2 个异常值
- s_auc_SLURRY_FLOW_LINE_A: 1 个异常值
- s_auc_SLURRY_FLOW_LINE_B: 2 个异常值
- s_auc_SLURRY_FLOW_LINE_C: 3 个异常值
- s_auc_WAFER_ROTATION: 50 个异常值
- s_auc_STAGE_ROTATION: 21 个异常值
- s_auc_HEAD_ROTATION: 34 个异常值
- knn_0: 27 个异常值
- knn_1: 20 个异常值
- knn_2: 20 个异常值
- knn_3: 30 个异常值
- knn_4: 6 个异常值
- knn_5: 19 个异常值
- knn_6: 13 个异常值
- knn_7: 10 个异常值
- knn_8: 20 个异常值
- knn_9: 30 个异常值
- knn_10: 17 个异常值
- len: 13 个异常值
- wt_max: 50 个异常值
- AVG_REMOVAL_RATE: 20 个异常值

**常量列**: f_median_SLURRY_FLOW_LINE_A, f_median_SLURRY_FLOW_LINE_B

### 2.3 描述性统计

```
           WAFER_ID  f_duration  f_USAGE_OF_BACKING_FILM  f_USAGE_OF_DRESSER  f_USAGE_OF_POLISHING_TABLE  f_USAGE_OF_DRESSER_TABLE  f_USAGE_OF_MEMBRANE  f_USAGE_OF_PRESSURIZED_SHEET  f_mean_PRESSURIZED_CHAMBER_PRESSURE  f_mean_MAIN_OUTER_AIR_BAG_PRESSURE  f_mean_CENTER_AIR_BAG_PRESSURE  f_mean_RETAINER_RING_PRESSURE  f_mean_RIPPLE_AIR_BAG_PRESSURE  f_mean_EDGE_AIR_BAG_PRESSURE  f_mean_DRESSING_WATER_STATUS  f_mean_SLURRY_FLOW_LINE_A  f_mean_SLURRY_FLOW_LINE_B  f_mean_SLURRY_FLOW_LINE_C  f_mean_WAFER_ROTATION  f_mean_STAGE_ROTATION  f_mean_HEAD_ROTATION  f_median_PRESSURIZED_CHAMBER_PRESSURE  f_median_MAIN_OUTER_AIR_BAG_PRESSURE  f_median_CENTER_AIR_BAG_PRESSURE  f_median_RETAINER_RING_PRESSURE  f_median_RIPPLE_AIR_BAG_PRESSURE  f_median_EDGE_AIR_BAG_PRESSURE  f_median_SLURRY_FLOW_LINE_A  f_median_SLURRY_FLOW_LINE_B  f_median_SLURRY_FLOW_LINE_C  f_median_WAFER_ROTATION  f_median_STAGE_ROTATION  f_median_HEAD_ROTATION  f_std_PRESSURIZED_CHAMBER_PRESSURE  f_std_MAIN_OUTER_AIR_BAG_PRESSURE  f_std_CENTER_AIR_BAG_PRESSURE  f_std_RETAINER_RING_PRESSURE  f_std_RIPPLE_AIR_BAG_PRESSURE  f_std_EDGE_AIR_BAG_PRESSURE  f_std_DRESSING_WATER_STATUS  f_std_SLURRY_FLOW_LINE_A  f_std_SLURRY_FLOW_LINE_B  f_std_SLURRY_FLOW_LINE_C  f_std_WAFER_ROTATION  f_std_STAGE_ROTATION  f_std_HEAD_ROTATION  f_auc_PRESSURIZED_CHAMBER_PRESSURE  f_auc_MAIN_OUTER_AIR_BAG_PRESSURE  f_auc_CENTER_AIR_BAG_PRESSURE  f_auc_RETAINER_RING_PRESSURE  f_auc_RIPPLE_AIR_BAG_PRESSURE  f_auc_EDGE_AIR_BAG_PRESSURE  f_auc_SLURRY_FLOW_LINE_A  f_auc_SLURRY_FLOW_LINE_B  f_auc_SLURRY_FLOW_LINE_C  f_auc_WAFER_ROTATION  f_auc_STAGE_ROTATION  f_auc_HEAD_ROTATION  f_is_first_run  f_last_mrr    s_duration  s_mean_PRESSURIZED_CHAMBER_PRESSURE  s_mean_MAIN_OUTER_AIR_BAG_PRESSURE  s_mean_CENTER_AIR_BAG_PRESSURE  s_mean_RETAINER_RING_PRESSURE  s_mean_RIPPLE_AIR_BAG_PRESSURE  s_mean_EDGE_AIR_BAG_PRESSURE  s_mean_DRESSING_WATER_STATUS  s_mean_SLURRY_FLOW_LINE_A  s_mean_SLURRY_FLOW_LINE_B  s_mean_SLURRY_FLOW_LINE_C  s_mean_WAFER_ROTATION  s_mean_STAGE_ROTATION  s_mean_HEAD_ROTATION  s_median_PRESSURIZED_CHAMBER_PRESSURE  s_median_MAIN_OUTER_AIR_BAG_PRESSURE  s_median_CENTER_AIR_BAG_PRESSURE  s_median_RETAINER_RING_PRESSURE  s_median_RIPPLE_AIR_BAG_PRESSURE  s_median_EDGE_AIR_BAG_PRESSURE  s_median_SLURRY_FLOW_LINE_A  s_median_SLURRY_FLOW_LINE_B  s_median_SLURRY_FLOW_LINE_C  s_median_WAFER_ROTATION  s_median_STAGE_ROTATION  s_median_HEAD_ROTATION  s_std_PRESSURIZED_CHAMBER_PRESSURE  s_std_MAIN_OUTER_AIR_BAG_PRESSURE  s_std_CENTER_AIR_BAG_PRESSURE  s_std_RETAINER_RING_PRESSURE  s_std_RIPPLE_AIR_BAG_PRESSURE  s_std_EDGE_AIR_BAG_PRESSURE  s_std_DRESSING_WATER_STATUS  s_std_SLURRY_FLOW_LINE_A  s_std_SLURRY_FLOW_LINE_B  s_std_SLURRY_FLOW_LINE_C  s_std_WAFER_ROTATION  s_std_STAGE_ROTATION  s_std_HEAD_ROTATION  s_auc_PRESSURIZED_CHAMBER_PRESSURE  s_auc_MAIN_OUTER_AIR_BAG_PRESSURE  s_auc_CENTER_AIR_BAG_PRESSURE  s_auc_RETAINER_RING_PRESSURE  s_auc_RIPPLE_AIR_BAG_PRESSURE  s_auc_EDGE_AIR_BAG_PRESSURE  s_auc_SLURRY_FLOW_LINE_A  s_auc_SLURRY_FLOW_LINE_B  s_auc_SLURRY_FLOW_LINE_C  s_auc_WAFER_ROTATION  s_auc_STAGE_ROTATION  s_auc_HEAD_ROTATION       knn_0       knn_1       knn_2       knn_3       knn_4       knn_5       knn_6       knn_7       knn_8       knn_9      knn_10         len        wt_max  mode_knn_0  mode_knn_1  mode_knn_2  AVG_REMOVAL_RATE
count  7.930000e+02  793.000000               793.000000          793.000000                  793.000000                793.000000           793.000000                    793.000000                           793.000000                          793.000000                      793.000000                     793.000000                      793.000000                    793.000000                    793.000000               7.930000e+02               7.930000e+02                 793.000000             793.000000             793.000000            793.000000                             793.000000                            793.000000                        793.000000                       793.000000                        793.000000                      793.000000                 7.930000e+02                 7.930000e+02                   793.000000               793.000000               793.000000              793.000000                          793.000000                         793.000000                     793.000000                    793.000000                     793.000000                   793.000000                   793.000000                793.000000              7.930000e+02                793.000000            793.000000            793.000000           793.000000                          793.000000                         793.000000                     793.000000                    793.000000                     793.000000                   793.000000                793.000000                793.000000                793.000000            793.000000            793.000000           793.000000      793.000000  793.000000    793.000000                           793.000000                          793.000000                      793.000000                     793.000000                      793.000000                    793.000000                    793.000000                 793.000000                 793.000000                 793.000000             793.000000             793.000000            793.000000                             793.000000                            793.000000                        793.000000                       793.000000                        793.000000                      793.000000                   793.000000                   793.000000                   793.000000               793.000000               793.000000              793.000000                          793.000000                         793.000000                     793.000000                    793.000000                     793.000000                   793.000000                   793.000000                793.000000                793.000000                793.000000            793.000000            793.000000           793.000000                          793.000000                         793.000000                     793.000000                  7.930000e+02                     793.000000                   793.000000                793.000000                793.000000                793.000000            793.000000            793.000000         7.930000e+02  793.000000  793.000000  793.000000  793.000000  793.000000  793.000000  793.000000  793.000000  793.000000  793.000000  793.000000  793.000000    793.000000  793.000000  793.000000  793.000000        793.000000
mean   8.521252e+08  113.346774              4780.959437          395.564897                  165.846527               3493.940731            56.691219                   1434.287831                            77.954733                          269.168029                       71.708048                    1468.428086                        9.937060                     48.508871                      0.315259               2.222222e+00               9.090909e-01                 435.733493              19.722765              38.895555            160.781778                              77.964031                            269.320555                         71.833622                      1450.096721                          9.944429                       48.483893                 2.222222e+00                 9.090909e-01                   440.676923                31.230535                 2.801404              158.743001                           31.765484                         116.594105                      31.009893                   1605.817799                       4.323793                    21.891540                     0.277281                  7.033305              4.193522e-02                192.295793             15.902416             90.849715             6.720009                        11450.022546                       35318.604703                    9363.415373                 286627.426345                    1329.907248                  6688.692058                981.526144                166.541904              57081.967171           2363.844585           9882.426016         29376.138086        0.501892   73.069818    182.113485                            48.887132                          153.581990                       40.372682                    1167.284900                        5.790218                     28.655963                      0.389117                   3.912511                   0.639597                 236.343545              16.447086              64.729941            160.376173                              53.060109                            184.839344                         48.824677                       988.790164                          6.858936                       32.757461                     1.567798                     0.639688                   297.507945                 2.083199                 1.981151              158.936696                           48.887132                         153.581990                      40.372682                   1167.284900                       5.790218                    28.655963                     0.389117                  3.912511                  0.639597                236.343545             16.447086             64.729941           160.376173                         7978.110107                       25183.608054                    6623.830260                  1.878581e+05                     948.846959                  4691.847222                613.055812                103.282314              38610.257646           2582.287725          10155.874058         2.777081e+04   72.938992   73.096476   73.086245   72.742423   73.078174   72.980652   73.216286   73.186349   73.244829   73.349510   73.647344  341.983607     49.221963   80.560782   80.172133   80.209503         73.079242
std    2.800101e+09    5.342029              2922.806032          219.862635                   94.750982                489.576848            34.657779                    876.841809                             2.225112                            3.070096                        1.791492                      17.169327                        0.151275                      0.984962                      0.464912               1.628404e-15               8.532984e-16                   6.640162               4.660791              18.331714              5.770638                               2.234848                              2.940091                          1.777335                         5.467721                          0.149931                        0.965933                 4.443695e-16                 1.110924e-16                     6.525520                10.283828                17.653372                5.837823                            1.433550                           3.307193                       1.118371                     95.338884                       0.136783                     0.825182                     0.188801                  0.567848              3.531041e-02                  7.740119              2.955096             15.473229             1.512478                          533.438825                        1396.226288                     405.298785                  11387.733248                      52.153777                   289.032729                107.970486                  6.263687               2603.085972            514.501637           2094.885030          1477.281043        0.500312    6.668845    494.710627                            32.759386                          103.569467                       27.163552                     776.216986                        3.901216                     19.181852                      0.433997                   2.380163                   0.392557                 150.767303               2.025135               8.021730              0.574417                              38.189735                            133.054019                         34.833111                       693.078916                          4.935632                       23.229225                     1.020876                     0.415393                   206.445586                 6.927306                15.993219                1.498578                           32.759386                         103.569467                      27.163552                    776.216986                       3.901216                    19.181852                     0.433997                  2.380163                  0.392557                150.767303              2.025135              8.021730             0.574417                         5835.017087                       18883.370116                    4979.382970                  1.312945e+05                     707.941119                  3467.799182                379.930997                 69.269578              28659.111448            307.779096           1198.698731         4.035648e+04    6.587421    6.642062    6.774936    6.904720    6.986868    6.752433    6.711105    6.771012    6.825075    6.686493    6.251970    7.974928    550.652854    8.948750    8.928700    9.107658          6.671836
min   -4.230161e+09   99.000000                29.166667            7.777778                   11.111111               2664.750000             0.345850                      8.750000                            76.393503                          246.324590                       59.951332                    1434.842500                        8.923621                     41.788376                      0.000000               2.222222e+00               9.090909e-01                 411.574775               0.000000               0.000000              0.000000                              76.190476                            250.800000                         61.250000                      1427.400000                          9.090909                       42.424242                 2.222222e+00                 9.090909e-01                   414.400000                 0.000000                 0.000000                0.000000                           24.924675                          95.502325                      24.694877                    831.674531                       3.530740                    17.057513                     0.000000                  3.035773              0.000000e+00                105.846034              0.000000              0.000000             1.476395                        10284.521905                       31570.999800                    7963.262500                 202406.741550                    1193.769977                  5637.027273                407.083333                136.363636              48769.467600              0.000000              0.000000          1590.400000        0.000000   53.426550    152.000000                             0.000000                            0.000000                        0.000000                       0.000000                        0.000000                      0.000000                      0.000000                   0.000000                   0.000000                   0.000000              12.130776              44.379457            148.828743                               0.000000                              0.000000                          0.000000                         0.000000                          0.000000                        0.000000                     0.000000                     0.000000                     0.000000                 0.000000                 0.000000              156.800000                            0.000000                           0.000000                       0.000000                      0.000000                       0.000000                     0.000000                     0.000000                  0.000000                  0.000000                  0.000000             12.130776             44.379457           148.828743                            0.000000                           0.000000                       0.000000                  0.000000e+00                       0.000000                     0.000000                  0.000000                  0.000000                  0.000000           2439.971279           9343.900133         2.435039e+04   53.426550   53.426550   53.694300   53.647350   53.694300   53.426550   53.647350   53.426550   53.647350   53.647350   53.647350  308.000000     14.000000   57.129900   57.129900   56.509050         53.426550
25%   -9.031702e+08  110.000000              2245.833333          218.888889                   83.703704               2998.000000            26.630435                    673.750000                            77.142857                          269.460000                       72.015482                    1452.018750                        9.952936                     48.606602                      0.000000               2.222222e+00               9.090909e-01                 431.200000              19.646107              32.289882            160.740496                              77.142857                            270.000000                         72.187500                      1446.900000                          9.954545                       48.484848                 2.222222e+00                 9.090909e-01                   436.800000                34.651163                 0.000000              156.800000                           31.393191                         115.832559                      30.879310                   1552.238806                       4.293479                    21.796224                     0.000000                  6.615626              2.226571e-16                189.019938             16.498539             92.008595             6.621095                        11072.051191                       34311.741601                    9107.878282                 279342.562219                    1292.310341                  6514.230303                853.632917                163.030000              55271.904801           2487.437442           9891.260131         28826.664000        0.000000   69.684900    157.000000                             0.000000                            0.000000                        0.000000                       0.000000                        0.000000                      0.000000                      0.058065                   0.551994                   0.023310                   9.392405              15.842278              62.912449            160.164103                               0.000000                              0.000000                          0.000000                         0.000000                          0.000000                        0.000000                     0.000000                     0.000000                     0.000000                 0.000000                 0.000000              156.800000                            0.000000                           0.000000                       0.000000                      0.000000                       0.000000                     0.000000                     0.058065                  0.551994                  0.023310                  9.392405             15.842278             62.912449           160.164103                            0.000000                           0.000000                       0.000000                  0.000000e+00                       0.000000                     0.000000                 72.291667                  3.181818               1146.972401           2491.976744           9885.290724         2.519200e+04   69.795900   69.670350   69.598200   69.525150   69.447900   69.684900   69.696450   69.525150   69.893100   70.299450   70.189800  337.000000     16.000000   73.740450   73.740450   72.509550         69.684900
50%    1.332254e+09  113.000000              4740.000000          385.185185                  159.259259               3545.250000            56.205534                   1422.000000                            77.329932                          269.744681                       72.061353                    1469.553191                        9.964015                     48.670920                      0.000000               2.222222e+00               9.090909e-01                 436.215652              20.747886              36.936793            161.086792                              77.142857                            270.000000                         72.187500                      1450.800000                          9.954545                       48.787879                 2.222222e+00                 9.090909e-01                   439.600000                34.651163                 0.000000              160.000000                           31.875631                         117.326161                      31.275922                   1604.611448                       4.356221                    22.083794                     0.386387                  7.192748              6.577951e-02                191.003941             16.597667             93.000198             6.883022                        11312.741666                       35049.135601                    9318.602187                 285686.052675                    1321.625682                  6655.056061               1042.203819                166.363636              56842.690801           2507.474884          10009.262895         29400.000000        1.000000   74.015550    158.000000                            69.180581                          214.307692                       56.779627                    1654.817197                        8.066589                     40.685703                      0.126582                   5.279547                   0.903301                 325.780892              16.033180              63.909061            160.389744                              77.142857                            268.800000                         72.187500                      1446.900000                          9.954545                       48.484848                     2.222222                     0.909091                   434.000000                 0.000000                 0.000000              160.000000                           69.180581                         214.307692                      56.779627                   1654.817197                       8.066589                    40.685703                     0.126582                  5.279547                  0.903301                325.780892             16.033180             63.909061           160.389744                        11053.918571                       34397.939401                    9128.491094                  2.631671e+05                    1295.920114                  6528.181818                827.436736                142.727273              52157.785402           2511.818721          10004.912368         2.536373e+04   74.015550   73.989450   74.118600   73.900050   73.548150   74.039400   73.992900   74.304150   74.007300   74.444550   74.911500  342.000000     17.000000   82.129200   81.060000   81.776250         74.118600
75%    3.017015e+09  116.666000              7296.666667          592.222222                  250.370370               3927.000000            86.521739                   2189.000000                            78.488008                          269.920661                       72.101004                    1481.253191                        9.977489                     48.730408                      1.000000               2.222222e+00               9.090909e-01                 440.280374              21.765719              39.688883            161.387611                              78.571429                            270.000000                         72.187500                      1454.700000                         10.000000                       48.787879                 2.222222e+00                 9.090909e-01                   445.200000                34.651163                 0.000000              160.000000                           32.389834                         118.339793                      31.551992                   1658.941942                       4.397815                    22.296439                     0.417639                  7.401617              6.738630e-02                193.412744             16.681365             93.966506             7.111155                        11773.654524                       36265.059599                    9641.250000                 293100.457694                    1367.514045                  6876.363636               1068.307222                170.909091              58940.624398           2523.500814          10139.583816         30090.134400        1.000000   77.719050    159.333000                            71.607143                          223.292308                       59.152070                    1751.025949                        8.403263                     42.237469                      1.000000                   5.484784                   0.909091                 343.969231              16.233151              64.740216            160.591083                              77.619048                            270.000000                         72.187500                      1450.800000                          9.954545                       48.787879                     2.222222                     0.909091                   439.600000                 0.000000                 0.000000              160.000000                           71.607143                         223.292308                      59.152070                   1751.025949                       8.403263                    42.237469                     1.000000                  5.484784                  0.909091                343.969231             16.233151             64.740216           160.591083                        11473.871429                       35805.992400                    9499.739844                  2.785047e+05                    1347.166068                  6780.093485                853.764653                143.940000              55068.085800           2534.722907          10128.860789         2.555467e+04   77.313000   77.746950   77.871150   77.370150   77.939100   77.605800   77.790600   78.225450   77.997000   77.825400   78.013500  347.000000     17.666000   86.756850   86.812500   87.062100         77.719050
max    4.229774e+09  138.333000             10521.666667          771.481481                  344.444444               4304.500000           124.762846                   3156.500000                            90.952381                          270.011009                       73.089674                    1559.197059                       10.000000                     49.003190                      1.000000               2.222222e+00               9.090909e-01                 456.508738              26.517591             151.095306            164.771429                              90.952381                            270.000000                         73.125000                      1462.500000                         10.000000                       48.787879                 2.222222e+00                 9.090909e-01                   462.000000                34.883721               196.578947              160.000000                           38.333173                         124.655894                      33.406705                   2033.179618                       4.647881                    23.840114                     0.462264                  8.928235              1.394180e-01                232.842231             17.216417            112.932173            34.790929                        13608.880476                       40922.868599                   10878.873281                 326590.254921                    1536.160886                  7704.810757               1132.895694                190.000000              66273.241999           2590.073140          20052.167038         33437.867200        1.000000   88.702050  13900.000000                           120.027473                          405.984615                       91.778846                    2376.963057                       14.355186                     59.607615                      1.000000                  10.360762                   0.909091                 413.558282              28.729991             132.432502            163.302564                             146.190476                            498.000000                        120.625000                      2515.500000                         18.045455                       77.575758                     3.750000                     0.909091                   579.600000                34.651163               131.578947              160.000000                          120.027473                         405.984615                      91.778846                   2376.963057                      14.355186                    59.607615                     1.000000                 10.360762                  0.909091                413.558282             28.729991            132.432502           163.302564                        68440.330238                      236857.645199                   63332.199531                  1.291236e+06                    8772.024432                 42550.210454               2042.137917                805.908636             395584.698603           4425.735698          20064.509606         1.101622e+06   88.702050   88.702050   88.702050   86.887200   88.702050   86.887200   88.058700   86.773350   86.887200   86.773350   86.887200  367.000000  15225.334000   99.870450  101.464800  101.464800         88.702050
```

---

## 3. 模型训练与评估

### 3.1 模型对比

| 模型 | Train R² | Test R² | RMSE | MAE | CV R² (mean±std) |
|------|----------|---------|------|-----|------------------|
| RandomForest | 0.9779 | 0.8862 | 2.1499 | 1.6466 | 0.8525±0.0283 |
| XGBoost | 0.9997 | 0.8778 | 2.2277 | 1.7363 | 0.8406±0.0297 |
| LightGBM | 0.9956 | 0.8785 | 2.2218 | 1.7355 | 0.8441±0.0270 |
| GradientBoosting | 0.9979 | 0.8733 | 2.2686 | 1.7676 | 0.8422±0.0326 |
| SVR | 0.8059 | 0.7624 | 3.1068 | 2.3206 | 0.7434±0.0424 |


### 3.2 最佳模型详情

**模型名称**: RandomForest


**测试集性能**:
- R² Score: 0.8862
- RMSE: 2.1499
- MAE: 1.6466
- MSE: 4.6219


### 3.3 交叉验证结果


5折交叉验证结果:
- 平均 R²: 0.8525
- 标准差: 0.0283
- 稳定性: 良好


---

## 4. 特征重要性分析


### Top 10 重要特征

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | knn_0 | 0.7542 |
| 2 | knn_1 | 0.0227 |
| 3 | f_last_mrr | 0.0224 |
| 4 | knn_2 | 0.0167 |
| 5 | knn_3 | 0.0150 |
| 6 | knn_5 | 0.0128 |
| 7 | f_auc_SLURRY_FLOW_LINE_A | 0.0107 |
| 8 | knn_7 | 0.0088 |
| 9 | f_std_SLURRY_FLOW_LINE_A | 0.0088 |
| 10 | f_auc_PRESSURIZED_CHAMBER_PRESSURE | 0.0071 |


**解读**: 特征重要性反映了各特征对预测结果的影响程度，重要性越高的特征对模型预测的贡献越大。


---

## 5. 预测结果

未进行新数据预测。

---

## 6. 结论与建议

### 6.1 模型性能总结

模型性能等级：**良好** (R² = 0.8862)

模型性能良好，可以用于实际预测，但建议持续监控和优化。

### 6.2 使用建议

1. **模型部署**: 建议使用 RandomForest 模型进行生产环境部署
2. **数据要求**: 确保输入数据包含以下特征：WAFER_ID, f_duration, f_USAGE_OF_BACKING_FILM, f_USAGE_OF_DRESSER, f_USAGE_OF_POLISHING_TABLE, f_USAGE_OF_DRESSER_TABLE, f_USAGE_OF_MEMBRANE, f_USAGE_OF_PRESSURIZED_SHEET, f_mean_PRESSURIZED_CHAMBER_PRESSURE, f_mean_MAIN_OUTER_AIR_BAG_PRESSURE, f_mean_CENTER_AIR_BAG_PRESSURE, f_mean_RETAINER_RING_PRESSURE, f_mean_RIPPLE_AIR_BAG_PRESSURE, f_mean_EDGE_AIR_BAG_PRESSURE, f_mean_DRESSING_WATER_STATUS, f_mean_SLURRY_FLOW_LINE_A, f_mean_SLURRY_FLOW_LINE_B, f_mean_SLURRY_FLOW_LINE_C, f_mean_WAFER_ROTATION, f_mean_STAGE_ROTATION, f_mean_HEAD_ROTATION, f_median_PRESSURIZED_CHAMBER_PRESSURE, f_median_MAIN_OUTER_AIR_BAG_PRESSURE, f_median_CENTER_AIR_BAG_PRESSURE, f_median_RETAINER_RING_PRESSURE, f_median_RIPPLE_AIR_BAG_PRESSURE, f_median_EDGE_AIR_BAG_PRESSURE, f_median_SLURRY_FLOW_LINE_A, f_median_SLURRY_FLOW_LINE_B, f_median_SLURRY_FLOW_LINE_C, f_median_WAFER_ROTATION, f_median_STAGE_ROTATION, f_median_HEAD_ROTATION, f_std_PRESSURIZED_CHAMBER_PRESSURE, f_std_MAIN_OUTER_AIR_BAG_PRESSURE, f_std_CENTER_AIR_BAG_PRESSURE, f_std_RETAINER_RING_PRESSURE, f_std_RIPPLE_AIR_BAG_PRESSURE, f_std_EDGE_AIR_BAG_PRESSURE, f_std_DRESSING_WATER_STATUS, f_std_SLURRY_FLOW_LINE_A, f_std_SLURRY_FLOW_LINE_B, f_std_SLURRY_FLOW_LINE_C, f_std_WAFER_ROTATION, f_std_STAGE_ROTATION, f_std_HEAD_ROTATION, f_auc_PRESSURIZED_CHAMBER_PRESSURE, f_auc_MAIN_OUTER_AIR_BAG_PRESSURE, f_auc_CENTER_AIR_BAG_PRESSURE, f_auc_RETAINER_RING_PRESSURE, f_auc_RIPPLE_AIR_BAG_PRESSURE, f_auc_EDGE_AIR_BAG_PRESSURE, f_auc_SLURRY_FLOW_LINE_A, f_auc_SLURRY_FLOW_LINE_B, f_auc_SLURRY_FLOW_LINE_C, f_auc_WAFER_ROTATION, f_auc_STAGE_ROTATION, f_auc_HEAD_ROTATION, f_is_first_run, f_last_mrr, s_duration, s_mean_PRESSURIZED_CHAMBER_PRESSURE, s_mean_MAIN_OUTER_AIR_BAG_PRESSURE, s_mean_CENTER_AIR_BAG_PRESSURE, s_mean_RETAINER_RING_PRESSURE, s_mean_RIPPLE_AIR_BAG_PRESSURE, s_mean_EDGE_AIR_BAG_PRESSURE, s_mean_DRESSING_WATER_STATUS, s_mean_SLURRY_FLOW_LINE_A, s_mean_SLURRY_FLOW_LINE_B, s_mean_SLURRY_FLOW_LINE_C, s_mean_WAFER_ROTATION, s_mean_STAGE_ROTATION, s_mean_HEAD_ROTATION, s_median_PRESSURIZED_CHAMBER_PRESSURE, s_median_MAIN_OUTER_AIR_BAG_PRESSURE, s_median_CENTER_AIR_BAG_PRESSURE, s_median_RETAINER_RING_PRESSURE, s_median_RIPPLE_AIR_BAG_PRESSURE, s_median_EDGE_AIR_BAG_PRESSURE, s_median_SLURRY_FLOW_LINE_A, s_median_SLURRY_FLOW_LINE_B, s_median_SLURRY_FLOW_LINE_C, s_median_WAFER_ROTATION, s_median_STAGE_ROTATION, s_median_HEAD_ROTATION, s_std_PRESSURIZED_CHAMBER_PRESSURE, s_std_MAIN_OUTER_AIR_BAG_PRESSURE, s_std_CENTER_AIR_BAG_PRESSURE, s_std_RETAINER_RING_PRESSURE, s_std_RIPPLE_AIR_BAG_PRESSURE, s_std_EDGE_AIR_BAG_PRESSURE, s_std_DRESSING_WATER_STATUS, s_std_SLURRY_FLOW_LINE_A, s_std_SLURRY_FLOW_LINE_B, s_std_SLURRY_FLOW_LINE_C, s_std_WAFER_ROTATION, s_std_STAGE_ROTATION, s_std_HEAD_ROTATION, s_auc_PRESSURIZED_CHAMBER_PRESSURE, s_auc_MAIN_OUTER_AIR_BAG_PRESSURE, s_auc_CENTER_AIR_BAG_PRESSURE, s_auc_RETAINER_RING_PRESSURE, s_auc_RIPPLE_AIR_BAG_PRESSURE, s_auc_EDGE_AIR_BAG_PRESSURE, s_auc_SLURRY_FLOW_LINE_A, s_auc_SLURRY_FLOW_LINE_B, s_auc_SLURRY_FLOW_LINE_C, s_auc_WAFER_ROTATION, s_auc_STAGE_ROTATION, s_auc_HEAD_ROTATION, knn_0, knn_1, knn_2, knn_3, knn_4, knn_5, knn_6, knn_7, knn_8, knn_9, knn_10, len, wt_max, mode_knn_0, mode_knn_1, mode_knn_2
3. **模型更新**: 建议定期使用新数据重新训练模型，保持预测准确性
4. **异常监控**: 实时监控预测值，对异常情况进行告警

### 6.3 改进方向

- 增加训练数据量以提升模型泛化能力
- 尝试集成学习方法（模型融合）
- 进行超参数调优（网格搜索或贝叶斯优化）
- 考虑使用深度学习模型（如果数据量足够大）

---

## 附录

### A. 技术栈

- **编程语言**: Python 3.8+
- **机器学习框架**: scikit-learn, XGBoost, LightGBM
- **数据处理**: pandas, numpy
- **模型管理**: joblib

### B. 模型文件

训练好的模型已保存至 `models_saved/` 目录，可直接加载使用：

```python
import joblib
model = joblib.load('models_saved/RandomForest.pkl')
```

---

*报告由 Grinding Speed Prediction Agent 自动生成*
*基于轻量级大模型 + 传统ML模型的智能预测系统*
