def set_manual_initial_guess(initial_guess):
    if len(initial_guess) == 15:
        # ex12
        initial_guess[0] = 57.4277
        initial_guess[1] = 180.2340
        initial_guess[2] = 62.9725
        initial_guess[3] = 90.4587
        initial_guess[4] = 58.1567
        initial_guess[5] = 180.1121
        initial_guess[6] = 76.7069
        initial_guess[7] = 90.3785
        initial_guess[8] = 0.0873
        initial_guess[9] = 345.7385
        initial_guess[10] = 0.9655
        initial_guess[11] = 331.2906
        initial_guess[12] = 0.5268
        initial_guess[13] = 205.4436
        initial_guess[14] = 114.5614
    elif len(initial_guess) == 21:
        # example 1 toothbrush
        initial_guess[0] = 0.5000
        initial_guess[1] = 0.5000
        initial_guess[2] = 0.0000
        initial_guess[3] = 197.4190
        initial_guess[4] = 180.0000
        initial_guess[5] = 0.0000
        initial_guess[6] = 195.4840
        initial_guess[7] = 180.0000
        initial_guess[8] = 0.0000
        initial_guess[9] = 0.5000
        initial_guess[10] = 0.4520
        initial_guess[11] = 0.0000
        initial_guess[12] = 1.0000
        initial_guess[13] = 0.5000
        initial_guess[14] = 90.0000
        initial_guess[15] = 180.0000
        initial_guess[16] = 180.0000
        initial_guess[17] = 180.0000
        initial_guess[18] = 0.8820
        initial_guess[19] = 0.4730
        initial_guess[20] = 0.0000
        # better con
        initial_guess[0] = 0.5050
        initial_guess[1] = 0.4973
        initial_guess[2] = 1.8986
        initial_guess[3] = 197.9063
        initial_guess[4] = 180.0000
        initial_guess[5] = 0.1580
        initial_guess[6] = 197.9063
        initial_guess[7] = 180.0000
        initial_guess[8] = 0.0000
        initial_guess[9] = 0.5050
        initial_guess[10] = 0.4460
        initial_guess[11] = 0.0000
        initial_guess[12] = 0.9819
        initial_guess[13] = 0.5649
        initial_guess[14] = 89.2920
        initial_guess[15] = 177.3863
        initial_guess[16] = 183.7769
        initial_guess[17] = 179.9055
        initial_guess[18] = 0.8841
        initial_guess[19] = 0.4825
        initial_guess[20] = 1.8190
    elif len(initial_guess) == 65:
        # nook jan18 con
        initial_guess[0] = 0.0339
        initial_guess[1] = 120.3643
        initial_guess[2] = 79.1205
        initial_guess[3] = 80.6565
        initial_guess[4] = 45.1611
        initial_guess[5] = 215.9914
        initial_guess[6] = 41.4786
        initial_guess[7] = 88.8955
        initial_guess[8] = 48.5219
        initial_guess[9] = 66.9065
        initial_guess[10] = 45.0213
        initial_guess[11] = 84.3875
        initial_guess[12] = 133.4582
        initial_guess[13] = 248.7732
        initial_guess[14] = 53.2342
        initial_guess[15] = 0.5932
        initial_guess[16] = 68.0936
        initial_guess[17] = 74.0494
        initial_guess[18] = 129.3002
        initial_guess[19] = 166.6762
        initial_guess[20] = 78.3073
        initial_guess[21] = 131.8332
        initial_guess[22] = 303.0555
        initial_guess[23] = 77.7029
        initial_guess[24] = 80.4765
        initial_guess[25] = 80.9838
        initial_guess[26] = 103.1249
        initial_guess[27] = 261.3018
        initial_guess[28] = 83.0425
        initial_guess[29] = 71.5690
        initial_guess[30] = 52.2375
        initial_guess[31] = 77.4264
        initial_guess[32] = 106.2727
        initial_guess[33] = 46.0546
        initial_guess[34] = 230.3293
        initial_guess[35] = 91.4765
        initial_guess[36] = 121.1688
        initial_guess[37] = 53.5615
        initial_guess[38] = 84.6491
        initial_guess[39] = 291.2638
        initial_guess[40] = 102.9198
        initial_guess[41] = 0.6876
        initial_guess[42] = 112.5136
        initial_guess[43] = 70.3753
        initial_guess[44] = 45.2982
        initial_guess[45] = 169.8273
        initial_guess[46] = 77.6080
        initial_guess[47] = 95.8966
        initial_guess[48] = 282.4666
        initial_guess[49] = 73.8951
        initial_guess[50] = 71.0341
        initial_guess[51] = 80.3092
        initial_guess[52] = 89.2527
        initial_guess[53] = 44.0238
        initial_guess[54] = 232.7905
        initial_guess[55] = 70.3448
        initial_guess[56] = 134.6623
        initial_guess[57] = 81.2995
        initial_guess[58] = 128.8197
        initial_guess[59] = 92.1304
        initial_guess[60] = 257.3272
        initial_guess[61] = 83.4404
        initial_guess[62] = 68.9215
        initial_guess[63] = 62.6571
        initial_guess[64] = 87.1433

        # not sure what
        # initial_guess[0] = 0.0000
        # initial_guess[1] = 119.8618
        # initial_guess[2] = 79.3496
        # initial_guess[3] = 80.6535
        # initial_guess[4] = 45.0668
        # initial_guess[5] = 215.9729
        # initial_guess[6] = 41.4785
        # initial_guess[7] = 88.8719
        # initial_guess[8] = 48.4805
        # initial_guess[9] = 66.9064
        # initial_guess[10] = 45.0007
        # initial_guess[11] = 84.2298
        # initial_guess[12] = 133.4569
        # initial_guess[13] = 249.6770
        # initial_guess[14] = 52.5740
        # initial_guess[15] = 0.5947
        # initial_guess[16] = 68.0684
        # initial_guess[17] = 74.0989
        # initial_guess[18] = 129.4114
        # initial_guess[19] = 166.6631
        # initial_guess[20] = 78.5020
        # initial_guess[21] = 131.8235
        # initial_guess[22] = 303.0555
        # initial_guess[23] = 77.6609
        # initial_guess[24] = 82.1220
        # initial_guess[25] = 81.0129
        # initial_guess[26] = 103.0064
        # initial_guess[27] = 261.2797
        # initial_guess[28] = 83.0426
        # initial_guess[29] = 71.6041
        # initial_guess[30] = 52.3145
        # initial_guess[31] = 77.4284
        # initial_guess[32] = 106.3724
        # initial_guess[33] = 46.0826
        # initial_guess[34] = 230.3293
        # initial_guess[35] = 89.8930
        # initial_guess[36] = 121.2636
        # initial_guess[37] = 54.1481
        # initial_guess[38] = 83.1956
        # initial_guess[39] = 290.3230
        # initial_guess[40] = 103.4593
        # initial_guess[41] = 0.6889
        # initial_guess[42] = 112.5138
        # initial_guess[43] = 70.3137
        # initial_guess[44] = 45.4474
        # initial_guess[45] = 169.8342
        # initial_guess[46] = 77.7150
        # initial_guess[47] = 95.8950
        # initial_guess[48] = 282.4666
        # initial_guess[49] = 73.9050
        # initial_guess[50] = 71.1582
        # initial_guess[51] = 80.3145
        # initial_guess[52] = 89.2485
        # initial_guess[53] = 44.0161
        # initial_guess[54] = 232.7905
        # initial_guess[55] = 70.2896
        # initial_guess[56] = 134.6003
        # initial_guess[57] = 81.3114
        # initial_guess[58] = 128.8311
        # initial_guess[59] = 92.2336
        # initial_guess[60] = 257.3271
        # initial_guess[61] = 84.6367
        # initial_guess[62] = 68.9203
        # initial_guess[63] = 60.0141
        # initial_guess[64] = 86.8599

        #jan11 debug
        # initial_guess[0] = 0.0000
        # initial_guess[1] = 119.2233
        # initial_guess[2] = 79.1913
        # initial_guess[3] = 82.1532
        # initial_guess[4] = 45.0000
        # initial_guess[5] = 201.6030
        # initial_guess[6] = 25.0040
        # initial_guess[7] = 84.4423
        # initial_guess[8] = 45.2955
        # initial_guess[9] = 62.9916
        # initial_guess[10] = 45.0005
        # initial_guess[11] = 253.9681
        # initial_guess[12] = 136.5341
        # initial_guess[13] = 71.6130
        # initial_guess[14] = 86.2293
        # initial_guess[15] = 0.5334
        # initial_guess[16] = 53.0272
        # initial_guess[17] = 69.4819
        # initial_guess[18] = 128.1571
        # initial_guess[19] = 160.1242
        # initial_guess[20] = 130.9696
        # initial_guess[21] = 142.6677
        # initial_guess[22] = 308.3527
        # initial_guess[23] = 8.7069
        # initial_guess[24] = 72.9551
        # initial_guess[25] = 12.8496
        # initial_guess[26] = 85.8108
        # initial_guess[27] = 135.0648
        # initial_guess[28] = 315.5447
        # initial_guess[29] = 82.1624
        # initial_guess[30] = 56.7055
        # initial_guess[31] = 83.5931
        # initial_guess[32] = 93.8257
        # initial_guess[33] = 281.4611
        # initial_guess[34] = 89.6375
        # initial_guess[35] = 79.8733
        # initial_guess[36] = 83.7150
        # initial_guess[37] = 57.6925
        # initial_guess[38] = 61.7079
        # initial_guess[39] = 69.6770
        # initial_guess[40] = 69.1314
        # initial_guess[41] = 0.6309
        # initial_guess[42] = 89.2625
        # initial_guess[43] = 75.6865
        # initial_guess[44] = 93.7151
        # initial_guess[45] = 179.4431
        # initial_guess[46] = 107.6860
        # initial_guess[47] = 54.6466
        # initial_guess[48] = 233.7175
        # initial_guess[49] = 76.1709
        # initial_guess[50] = 48.0737
        # initial_guess[51] = 79.2122
        # initial_guess[52] = 111.4484
        # initial_guess[53] = 67.9492
        # initial_guess[54] = 252.0922
        # initial_guess[55] = 145.5668
        # initial_guess[56] = 70.0534
        # initial_guess[57] = 143.9920
        # initial_guess[58] = 130.8155
        # initial_guess[59] = 4.8890
        # initial_guess[60] = 177.1757
        # initial_guess[61] = 37.9873
        # initial_guess[62] = 105.2631
        # initial_guess[63] = 5.2063
        # initial_guess[64] = 73.1933

        # manual setting in progress
        # initial_guess[0] = 0.0000
        # initial_guess[1] = 115.6450
        # initial_guess[2] = 102.9030
        # initial_guess[3] = 79.1283
        # initial_guess[4] = 127.2245
        # initial_guess[5] = 91.5637
        # initial_guess[6] = 267.4301
        # initial_guess[7] = 69.6300
        # initial_guess[8] = 131.1290
        # initial_guess[9] = 81.4520
        # initial_guess[10] = 75.0000
        # initial_guess[11] = 305.8060
        # initial_guess[12] = 119.6770
        # initial_guess[13] = 118.0650
        # initial_guess[14] = 62.4802
        # initial_guess[15] = 0.5910
        # initial_guess[16] = 103.0650
        # initial_guess[17] = 45.0000
        # initial_guess[18] = 82.7420
        # initial_guess[19] = 200.3230
        # initial_guess[20] = 121.4520
        # initial_guess[21] = 57.1093
        # initial_guess[22] = 254.7118
        # initial_guess[23] = 64.3526
        # initial_guess[24] = 84.6172
        # initial_guess[25] = 75.0117
        # initial_guess[26] = 99.3060
        # initial_guess[27] = 66.7391
        # initial_guess[28] = 247.9885
        # initial_guess[29] = 79.3930
        # initial_guess[30] = 103.2282
        # initial_guess[31] = 81.8921
        # initial_guess[32] = 52.8527
        # initial_guess[33] = 318.1470
        # initial_guess[34] = 134.8267
        # initial_guess[35] = 69.1106
        # initial_guess[36] = 69.5331
        # initial_guess[37] = 85.2460
        # initial_guess[38] = 77.2504
        # initial_guess[39] = 145.1610
        # initial_guess[40] = 30.0345
        # initial_guess[41] = 0.7740
        # initial_guess[42] = 45.2985
        # initial_guess[43] = 0.1013
        # initial_guess[44] = 133.5771
        # initial_guess[45] = 223.9478
        # initial_guess[46] = 46.6655
        # initial_guess[47] = 282.7872
        # initial_guess[48] = 83.4870
        # initial_guess[49] = 90.8437
        # initial_guess[50] = 89.3944
        # initial_guess[51] = 75.2136
        # initial_guess[52] = 45.7324
        # initial_guess[53] = 131.7117
        # initial_guess[54] = 303.1143
        # initial_guess[55] = 67.6645
        # initial_guess[56] = 115.8352
        # initial_guess[57] = 75.2600
        # initial_guess[58] = 45.0002
        # initial_guess[59] = 128.5977
        # initial_guess[60] = 309.1667
        # initial_guess[61] = 89.2370
        # initial_guess[62] = 45.0064
        # initial_guess[63] = 66.5495
        # initial_guess[64] = 61.3190

        # ognookcon1 with turnbuckle
        # initial_guess[0] = 0.0000
        # initial_guess[1] = 120.8056
        # initial_guess[2] = 110.0537
        # initial_guess[3] = 114.2461
        # initial_guess[4] = 107.9273
        # initial_guess[5] = 251.9908
        # initial_guess[6] = 105.0260
        # initial_guess[7] = 128.6896
        # initial_guess[8] = 95.4338
        # initial_guess[9] = 148.1979
        # initial_guess[10] = 135.0000
        # initial_guess[11] = 325.3165
        # initial_guess[12] = 149.9999
        # initial_guess[13] = 327.2828
        # initial_guess[14] = 45.0196
        # initial_guess[15] = 0.5576
        # initial_guess[16] = 71.4641
        # initial_guess[17] = 76.1139
        # initial_guess[18] = 130.7908
        # initial_guess[19] = 171.9695
        # initial_guess[20] = 78.4421
        # initial_guess[21] = 252.5141
        # initial_guess[22] = 71.9220
        # initial_guess[23] = 73.9187
        # initial_guess[24] = 106.6168
        # initial_guess[25] = 76.4605
        # initial_guess[26] = 72.7248
        # initial_guess[27] = 288.2337
        # initial_guess[28] = 106.5097
        # initial_guess[29] = 71.1582
        # initial_guess[30] = 70.7956
        # initial_guess[31] = 74.3995
        # initial_guess[32] = 113.3280
        # initial_guess[33] = 347.8600
        # initial_guess[34] = 156.1721
        # initial_guess[35] = 81.4989
        # initial_guess[36] = 109.0112
        # initial_guess[37] = 63.9161
        # initial_guess[38] = 80.1923
        # initial_guess[39] = 273.1063
        # initial_guess[40] = 96.1109
        # initial_guess[41] = 0.6164
        # initial_guess[42] = 97.8962
        # initial_guess[43] = 75.9067
        # initial_guess[44] = 61.0079
        # initial_guess[45] = 176.4834
        # initial_guess[46] = 100.4536
        # initial_guess[47] = 276.8820
        # initial_guess[48] = 96.0664
        # initial_guess[49] = 78.0846
        # initial_guess[50] = 125.5942
        # initial_guess[51] = 79.3401
        # initial_guess[52] = 131.5117
        # initial_guess[53] = 269.5006
        # initial_guess[54] = 89.2455
        # initial_guess[55] = 72.9056
        # initial_guess[56] = 56.2311
        # initial_guess[57] = 72.8143
        # initial_guess[58] = 65.8258
        # initial_guess[59] = 272.9753
        # initial_guess[60] = 94.1782
        # initial_guess[61] = 77.5245
        # initial_guess[62] = 86.4454
        # initial_guess[63] = 55.0213
        # initial_guess[64] = 87.8650

        # automatically computed with fix4 off
        # initial_guess[0] = 0.7700
        # initial_guess[1] = 124.5906
        # initial_guess[2] = 92.5687
        # initial_guess[3] = 67.6086
        # initial_guess[4] = 129.0738
        # initial_guess[5] = 140.1466
        # initial_guess[6] = 291.4460
        # initial_guess[7] = 101.2589
        # initial_guess[8] = 115.2215
        # initial_guess[9] = 61.5350
        # initial_guess[10] = 131.5681
        # initial_guess[11] = 104.1867
        # initial_guess[12] = 67.3045
        # initial_guess[13] = 256.1198
        # initial_guess[14] = 99.1985
        # initial_guess[15] = 0.6386
        # initial_guess[16] = 108.5091
        # initial_guess[17] = 76.2555
        # initial_guess[18] = 49.6722
        # initial_guess[19] = 171.1398
        # initial_guess[20] = 124.6512
        # initial_guess[21] = 59.6261
        # initial_guess[22] = 242.8860
        # initial_guess[23] = 78.0742
        # initial_guess[24] = 86.8202
        # initial_guess[25] = 77.8997
        # initial_guess[26] = 115.6371
        # initial_guess[27] = 123.8939
        # initial_guess[28] = 305.1013
        # initial_guess[29] = 76.3721
        # initial_guess[30] = 117.1860
        # initial_guess[31] = 76.9319
        # initial_guess[32] = 63.1273
        # initial_guess[33] = 303.2238
        # initial_guess[34] = 127.0878
        # initial_guess[35] = 81.4276
        # initial_guess[36] = 62.0553
        # initial_guess[37] = 59.4383
        # initial_guess[38] = 94.9376
        # initial_guess[39] = 127.1851
        # initial_guess[40] = 91.6629
        # initial_guess[41] = 0.6171
        # initial_guess[42] = 78.2097
        # initial_guess[43] = 74.7713
        # initial_guess[44] = 131.4933
        # initial_guess[45] = 175.5283
        # initial_guess[46] = 64.8477
        # initial_guess[47] = 83.9861
        # initial_guess[48] = 265.5880
        # initial_guess[49] = 75.8476
        # initial_guess[50] = 62.5100
        # initial_guess[51] = 78.2969
        # initial_guess[52] = 91.5615
        # initial_guess[53] = 278.7572
        # initial_guess[54] = 99.9627
        # initial_guess[55] = 74.7532
        # initial_guess[56] = 106.5192
        # initial_guess[57] = 75.3082
        # initial_guess[58] = 77.2706
        # initial_guess[59] = 174.3553
        # initial_guess[60] = 357.7554
        # initial_guess[61] = 83.9959
        # initial_guess[62] = 85.7894
        # initial_guess[63] = 61.8075
        # initial_guess[64] = 97.3941

        # nookcon5
        # initial_guess[0] = 0.8090
        # initial_guess[1] = 115.6340
        # initial_guess[2] = 88.1399
        # initial_guess[3] = 83.6026
        # initial_guess[4] = 75.6697
        # initial_guess[5] = 8.3893
        # initial_guess[6] = 201.2898
        # initial_guess[7] = 96.4361
        # initial_guess[8] = 52.8746
        # initial_guess[9] = 106.9762
        # initial_guess[10] = 113.3339
        # initial_guess[11] = 65.8315
        # initial_guess[12] = 45.3505
        # initial_guess[13] = 235.2992
        # initial_guess[14] = 109.1660
        # initial_guess[15] = 0.5959
        # initial_guess[16] = 90.7617
        # initial_guess[17] = 61.6020
        # initial_guess[18] = 85.0731
        # initial_guess[19] = 182.3096
        # initial_guess[20] = 97.5775
        # initial_guess[21] = 48.7546
        # initial_guess[22] = 230.9982
        # initial_guess[23] = 79.6131
        # initial_guess[24] = 125.3252
        # initial_guess[25] = 75.4126
        # initial_guess[26] = 129.9695
        # initial_guess[27] = 346.9672
        # initial_guess[28] = 175.5909
        # initial_guess[29] = 77.6420
        # initial_guess[30] = 94.2916
        # initial_guess[31] = 74.5982
        # initial_guess[32] = 52.6880
        # initial_guess[33] = 330.9929
        # initial_guess[34] = 141.3719
        # initial_guess[35] = 95.9211
        # initial_guess[36] = 54.2968
        # initial_guess[37] = 77.0171
        # initial_guess[38] = 45.1548
        # initial_guess[39] = 219.2733
        # initial_guess[40] = 128.7890
        # initial_guess[41] = 0.7400
        # initial_guess[42] = 86.3920
        # initial_guess[43] = 61.9336
        # initial_guess[44] = 108.5860
        # initial_guess[45] = 183.0017
        # initial_guess[46] = 85.9546
        # initial_guess[47] = 216.8284
        # initial_guess[48] = 39.2387
        # initial_guess[49] = 80.7610
        # initial_guess[50] = 121.1695
        # initial_guess[51] = 71.8756
        # initial_guess[52] = 71.7109
        # initial_guess[53] = 189.9398
        # initial_guess[54] = 19.9799
        # initial_guess[55] = 73.8179
        # initial_guess[56] = 113.1472
        # initial_guess[57] = 76.0285
        # initial_guess[58] = 53.5195
        # initial_guess[59] = 146.3612
        # initial_guess[60] = 320.0796
        # initial_guess[61] = 88.2959
        # initial_guess[62] = 46.0113
        # initial_guess[63] = 69.8971
        # initial_guess[64] = 60.6243

        # for reading nook
        # initial_guess[0] = 6.610696436837316e-05
        # initial_guess[1] = 113.1257553100586
        # initial_guess[2] = 88.56434631347656
        # initial_guess[3] = 84.6656494140625
        # initial_guess[4] = 64.87377166748047
        # initial_guess[5] = 11.847233772277832
        # initial_guess[6] = 204.85267639160156
        # initial_guess[7] = 69.0662612915039
        # initial_guess[8] = 49.36127853393555
        # initial_guess[9] = 94.89861297607422
        # initial_guess[10] = 73.7738037109375
        # initial_guess[11] = 71.4615707397461
        # initial_guess[12] = 83.70201110839844
        # initial_guess[15] = 0.5907921195030212
        # initial_guess[16] = 108.08573913574219
        # initial_guess[17] = 178.4336395263672
        # initial_guess[18] = 117.4176025390625
        # initial_guess[19] = 95.86280822753906
        # initial_guess[20] = 110.40019989013672
        # initial_guess[21] = 300.4462890625
        # initial_guess[22] = 159.01217651367188
        # initial_guess[23] = 46.760189056396484
        # initial_guess[24] = 133.93923950195312
        # initial_guess[25] = 54.83855438232422
        # initial_guess[26] = 134.328857421875
        # initial_guess[27] = 350.27398681640625
        # initial_guess[28] = 180.4579620361328
        # initial_guess[29] = 44.19963455200195
        # initial_guess[30] = 92.05503845214844
        # initial_guess[31] = 35.33287811279297
        # initial_guess[32] = 112.51461029052734
        # initial_guess[33] = 276.7415771484375
        # initial_guess[34] = 98.21566772460938
        # initial_guess[35] = 72.42618560791016
        # initial_guess[36] = 96.00736999511719
        # initial_guess[37] = 87.61284637451172
        # initial_guess[38] = 96.93506622314453
        # initial_guess[39] = 191.0486297607422
        # initial_guess[40] = 98.17931365966797
        # initial_guess[41] = 0.4787823557853699
        # initial_guess[42] = 66.94056701660156
        # initial_guess[43] = 76.46405029296875
        # initial_guess[44] = 120.96075439453125
        # initial_guess[45] = 188.82730102539062
        # initial_guess[46] = 135.0
        # initial_guess[47] = 196.42893981933594
        # initial_guess[48] = 20.372020721435547
        # initial_guess[49] = 50.8029670715332
        # initial_guess[50] = 101.29724884033203
        # initial_guess[51] = 49.12799072265625
        # initial_guess[52] = 113.20010375976562
        # initial_guess[53] = 184.28517150878906
        # initial_guess[54] = 337.71258544921875
        # initial_guess[55] = 116.5127182006836
        # initial_guess[56] = 131.02735900878906
        # initial_guess[57] = 94.74420928955078
        # initial_guess[58] = 115.89964294433594
        # initial_guess[59] = 103.78630065917969
        # initial_guess[60] = 291.99713134765625
        # initial_guess[61] = 70.8627700805664
        # initial_guess[62] = 56.932151794433594
        # initial_guess[63] = 80.98493957519531
        # initial_guess[64] = 103.8512191772461

        # initial_guess[0] = 0.0000
        # initial_guess[1] = 119.2823
        # initial_guess[2] = 99.6827
        # initial_guess[3] = 99.9525
        # initial_guess[4] = 134.7961
        # initial_guess[5] = 198.3474
        # initial_guess[6] = 24.8877
        # initial_guess[7] = 27.7698
        # initial_guess[8] = 133.2725
        # initial_guess[9] = 4.9455
        # initial_guess[10] = 56.7061
        # initial_guess[11] = 353.3894
        # initial_guess[12] = 126.9084
        # initial_guess[13] = 277.7349
        # initial_guess[14] = 113.9049
        # initial_guess[15] = 0.5106
        # initial_guess[16] = 134.9839
        # initial_guess[17] = 145.8862
        # initial_guess[18] = 46.9579
        # initial_guess[19] = 76.4129
        # initial_guess[20] = 129.6790
        # initial_guess[21] = 283.4912
        # initial_guess[22] = 120.8332
        # initial_guess[23] = 91.0870
        # initial_guess[24] = 96.5586
        # initial_guess[25] = 69.8342
        # initial_guess[26] = 106.5245
        # initial_guess[27] = 18.2232
        # initial_guess[28] = 196.9612
        # initial_guess[29] = 70.2348
        # initial_guess[30] = 124.5427
        # initial_guess[31] = 76.2898
        # initial_guess[32] = 79.3533
        # initial_guess[33] = 177.9989
        # initial_guess[34] = 357.8105
        # initial_guess[35] = 60.9953
        # initial_guess[36] = 46.2018
        # initial_guess[37] = 63.3199
        # initial_guess[38] = 84.8313
        # initial_guess[39] = 189.7604
        # initial_guess[40] = 34.3306
        # initial_guess[41] = 0.6201
        # initial_guess[42] = 80.3285
        # initial_guess[43] = 251.2165
        # initial_guess[44] = 92.3044
        # initial_guess[45] = 1.7611
        # initial_guess[46] = 85.3711
        # initial_guess[47] = 143.3316
        # initial_guess[48] = 317.1832
        # initial_guess[49] = 16.1958
        # initial_guess[50] = 50.0602
        # initial_guess[51] = 5.0032
        # initial_guess[52] = 45.0000
        # initial_guess[53] = 150.2800
        # initial_guess[54] = 335.6634
        # initial_guess[55] = 86.6065
        # initial_guess[56] = 127.5002
        # initial_guess[57] = 87.0793
        # initial_guess[58] = 80.3387
        # initial_guess[59] = 4.8496
        # initial_guess[60] = 199.8858
        # initial_guess[61] = 76.2540
        # initial_guess[62] = 53.5430
        # initial_guess[63] = 71.2428
        # initial_guess[64] = 58.7556

        # initial_guess[0] = 0.7254
        # initial_guess[1] = 119.1512
        # initial_guess[2] = 82.5337
        # initial_guess[3] = 66.1824
        # initial_guess[4] = 131.6686
        # initial_guess[5] = 50.3113
        # initial_guess[6] = 211.7935
        # initial_guess[7] = 68.9231
        # initial_guess[8] = 131.3844
        # initial_guess[9] = 34.8617
        # initial_guess[10] = 45.0000
        # initial_guess[11] = 316.5452
        # initial_guess[12] = 140.8152
        # initial_guess[13] = 102.7183
        # initial_guess[14] = 124.0844
        # initial_guess[15] = 0.5696
        # initial_guess[16] = 45.0000
        # initial_guess[17] = 237.1067
        # initial_guess[18] = 134.0320
        # initial_guess[19] = 166.4457
        # initial_guess[20] = 134.8679
        # initial_guess[21] = 19.4581
        # initial_guess[22] = 215.3002
        # initial_guess[23] = 89.4604
        # initial_guess[24] = 102.7372
        # initial_guess[25] = 74.1948
        # initial_guess[26] = 87.2693
        # initial_guess[27] = 287.4310
        # initial_guess[28] = 111.8180
        # initial_guess[29] = 72.3620
        # initial_guess[30] = 105.5659
        # initial_guess[31] = 77.5560
        # initial_guess[32] = 85.8446
        # initial_guess[33] = 158.6510
        # initial_guess[34] = 337.1625
        # initial_guess[35] = 78.3366
        # initial_guess[36] = 78.2421
        # initial_guess[37] = 70.6176
        # initial_guess[38] = 63.0379
        # initial_guess[39] = 300.1195
        # initial_guess[40] = 88.2262
        # initial_guess[41] = 0.5757
        # initial_guess[42] = 45.6657
        # initial_guess[43] = 238.8725
        # initial_guess[44] = 134.6093
        # initial_guess[45] = 165.7247
        # initial_guess[46] = 45.0026
        # initial_guess[47] = 78.5968
        # initial_guess[48] = 274.1031
        # initial_guess[49] = 59.4895
        # initial_guess[50] = 80.5671
        # initial_guess[51] = 74.9059
        # initial_guess[52] = 134.5769
        # initial_guess[53] = 302.6178
        # initial_guess[54] = 124.3762
        # initial_guess[55] = 77.4235
        # initial_guess[56] = 113.8621
        # initial_guess[57] = 68.1629
        # initial_guess[58] = 102.1328
        # initial_guess[59] = 196.0177
        # initial_guess[60] = 20.5387
        # initial_guess[61] = 85.4553
        # initial_guess[62] = 94.5461
        # initial_guess[63] = 66.8535
        # initial_guess[64] = 119.3983
    elif len(initial_guess) == 53:
        initial_guess[0] = 0.0005
        initial_guess[1] = 112.0272
        initial_guess[2] = 90.6209
        initial_guess[3] = 77.9186
        initial_guess[4] = 70.9591
        initial_guess[5] = 292.2623
        initial_guess[6] = 106.9151
        initial_guess[7] = 36.7462
        initial_guess[8] = 45.0000
        initial_guess[9] = 0.0001
        initial_guess[10] = 134.1176
        initial_guess[11] = 284.5912
        initial_guess[12] = 43.7865
        initial_guess[13] = 325.0573
        initial_guess[14] = 132.2562
        initial_guess[15] = 0.2895
        initial_guess[16] = 104.3578
        initial_guess[17] = 132.2485
        initial_guess[18] = 78.9995
        initial_guess[19] = 177.5102
        initial_guess[20] = 166.2882
        initial_guess[21] = 68.2902
        initial_guess[22] = 107.4713
        initial_guess[23] = 135.0000
        initial_guess[24] = 98.0299
        initial_guess[25] = 280.1977
        initial_guess[26] = 94.2813
        initial_guess[27] = 68.4545
        initial_guess[28] = 52.1829
        initial_guess[29] = 52.3407
        initial_guess[30] = 271.6885
        initial_guess[31] = 84.7379
        initial_guess[32] = 0.7107
        initial_guess[33] = 94.3938
        initial_guess[34] = 10.1342
        initial_guess[35] = 75.8387
        initial_guess[36] = 175.2480
        initial_guess[37] = 159.6673
        initial_guess[38] = 62.3201
        initial_guess[39] = 91.5339
        initial_guess[40] = 113.8597
        initial_guess[41] = 240.1190
        initial_guess[42] = 54.6525
        initial_guess[43] = 68.7622
        initial_guess[44] = 59.1201
        initial_guess[45] = 60.8106
        initial_guess[46] = 45.0024
        initial_guess[47] = 298.3102
        initial_guess[48] = 147.2946
        initial_guess[49] = 96.3557
        initial_guess[50] = 49.1829
        initial_guess[51] = 87.3737
        initial_guess[52] = 45.0024
    elif len(initial_guess) == 47:
        # for smaller nook debug - paramturnbuckle
        # initial_guess[0] = 0
        # initial_guess[1] = 97.258
        # initial_guess[2] = 96.989
        # initial_guess[3] = 100.538
        # initial_guess[4] = 135
        # initial_guess[5] = 0
        # initial_guess[6] = 203.226
        # initial_guess[7] = 90.323
        # initial_guess[8] = 135
        # initial_guess[9] = 53.226
        # initial_guess[10] = 90
        # initial_guess[11] = 0
        # initial_guess[12] = 100.645
        # initial_guess[15] = 0.608
        # initial_guess[16] = 92.419
        # initial_guess[17] = 126.452
        # initial_guess[18] = 90
        # initial_guess[19] = 180
        # initial_guess[20] = 172.742
        # initial_guess[21] = 75
        # initial_guess[22] = 100.161
        # initial_guess[23] = 45
        # initial_guess[24] = 176.129
        # initial_guess[25] = 0
        # initial_guess[26] = 77.419
        # initial_guess[27] = 135
        # initial_guess[32] = 0.355
        # initial_guess[33] = 84.194
        # initial_guess[34] = 45.161
        # initial_guess[35] = 90
        # initial_guess[36] = 180
        # initial_guess[37] = 169.839
        # initial_guess[38] = 90
        # initial_guess[39] = 94.355
        # initial_guess[40] = 45
        # initial_guess[41] = 0
        # initial_guess[42] = 181.935
        # initial_guess[43] = 85.484
        # initial_guess[44] = 135

        # for smaller nook debug - turnbuckle
        # initial_guess[0 +15] = 0.4128
        # initial_guess[1 +15] = 87.5670
        # initial_guess[2 +15] = 135.2224
        # initial_guess[3 +15] = 97.3393
        # initial_guess[4 +15] = 169.1422
        # initial_guess[5 +15] = 169.2852
        # initial_guess[6 +15] = 89.8008
        # initial_guess[7 +15] = 105.3830
        # initial_guess[8 +15] = 71.7562
        # initial_guess[9 +15] = 336.3590
        # initial_guess[10 +15] = 151.4841
        # initial_guess[11 +15] = 109.7722
        # initial_guess[12 +15] = 94.8930
        # initial_guess[13 +15] = 59.4412
        # initial_guess[14 +15] = 109.9487
        # # initial_guess[15] = 169.7599
        # # initial_guess[16] = 83.3065
        # initial_guess[0 +32] =0.5430
        # initial_guess[1 +32] =103.8405
        # initial_guess[2 +32] =63.9669
        # initial_guess[3 +32] =58.0829
        # initial_guess[4 +32] =174.0649
        # initial_guess[5 +32] =196.8497
        # initial_guess[6 +32] =108.5059
        # initial_guess[7 +32] =101.4877
        # initial_guess[8 +32] =89.9288
        # initial_guess[9 +32] =54.2901
        # initial_guess[10 +32] =235.2420
        # initial_guess[11 +32] =105.0321
        # initial_guess[12 +32] =91.7075
        # initial_guess[13 +32] =51.6503
        # initial_guess[14 +32] =120.4727
        # # initial_guess[15] = 228.0867
        # # initial_guess[16] = 69.1875
        # initial_guess[0] = 0
        # initial_guess[1] = 107.2223
        # initial_guess[2] = 90.5642
        # initial_guess[3] = 83.5223
        # initial_guess[4] = 65.3693
        # initial_guess[5] = 93.6590
        # initial_guess[6] = 273.2819
        # initial_guess[7] = 74.4113
        # initial_guess[8] = 62.0887
        # initial_guess[9] = 36.0666
        # initial_guess[10] = 110.2848
        # initial_guess[11] = 208.2825
        # initial_guess[12] = 69.8134

        # newtb_con2 saved params
        initial_guess[0] = 0.0051
        initial_guess[1] = 111.1079
        initial_guess[2] = 82.4721
        initial_guess[3] = 83.0332
        initial_guess[4] = 102.6682
        initial_guess[5] = 90.8730
        initial_guess[6] = 273.2837
        initial_guess[7] = 70.4107
        initial_guess[8] = 60.1960
        initial_guess[9] = 36.1678
        initial_guess[10] = 73.4584
        initial_guess[11] = 195.3218
        initial_guess[12] = 87.3668
        initial_guess[13] = 6.9147
        initial_guess[14] = 68.3095
        initial_guess[15] = 0.3482
        initial_guess[16] = 86.8125
        initial_guess[17] = 134.0182
        initial_guess[18] = 98.3803
        initial_guess[19] = 168.7239
        initial_guess[20] = 169.2494
        initial_guess[21] = 115.6163
        initial_guess[22] = 106.5949
        initial_guess[23] = 77.9532
        initial_guess[24] = 334.7823
        initial_guess[25] = 151.4865
        initial_guess[26] = 112.2146
        initial_guess[27] = 92.8364
        initial_guess[28] = 55.5186
        initial_guess[29] = 92.2890
        initial_guess[30] = 61.1814
        initial_guess[31] = 139.0457
        initial_guess[32] = 0.6796
        initial_guess[33] = 105.4963
        initial_guess[34] = 67.2261
        initial_guess[35] = 53.9535
        initial_guess[36] = 173.9114
        initial_guess[37] = 195.3005
        initial_guess[38] = 105.6473
        initial_guess[39] = 101.2787
        initial_guess[40] = 66.6367
        initial_guess[41] = 54.2924
        initial_guess[42] = 235.2418
        initial_guess[43] = 99.9887
        initial_guess[44] = 94.1952
        initial_guess[45] = 42.6195
        initial_guess[46] = 64.3174
    elif len(initial_guess) == 46:
        initial_guess[0] = 0.0346
        initial_guess[1] = 111.2625
        initial_guess[2] = 84.7900
        initial_guess[3] = 82.6859
        initial_guess[4] = 101.5921
        initial_guess[5] = 90.9814
        initial_guess[6] = 273.2838
        initial_guess[7] = 73.6291
        initial_guess[8] = 45.8130
        initial_guess[9] = 42.7503
        initial_guess[10] = 146.2296
        initial_guess[11] = 84.2819
        initial_guess[12] = 308.4365
        initial_guess[13] = 69.3936
        initial_guess[14] = 0.3491
        initial_guess[15] = 86.7151
        initial_guess[16] = 134.4658
        initial_guess[17] = 98.4255
        initial_guess[18] = 168.6917
        initial_guess[19] = 169.2473
        initial_guess[20] = 117.9416
        initial_guess[21] = 106.6696
        initial_guess[22] = 78.4035
        initial_guess[23] = 334.6281
        initial_guess[24] = 151.4868
        initial_guess[25] = 112.6030
        initial_guess[26] = 92.6998
        initial_guess[27] = 54.2863
        initial_guess[28] = 90.9627
        initial_guess[29] = 6.4941
        initial_guess[30] = 139.2054
        initial_guess[31] = 0.7021
        initial_guess[32] = 105.4752
        initial_guess[33] = 66.6201
        initial_guess[34] = 54.5969
        initial_guess[35] = 173.8637
        initial_guess[36] = 195.4457
        initial_guess[37] = 105.6275
        initial_guess[38] = 101.2823
        initial_guess[39] = 66.9931
        initial_guess[40] = 54.3227
        initial_guess[41] = 235.2418
        initial_guess[42] = 101.5039
        initial_guess[43] = 92.9893
        initial_guess[44] = 41.9935
        initial_guess[45] = 64.0297
    else:
        # for smaller nook
        initial_guess[0] = 0.444
        initial_guess[1] = 113.289
        initial_guess[2] = 95.806 #94.876
        initial_guess[3] = 94.252
        initial_guess[4] = 113.947
        initial_guess[5] = 352.764
        initial_guess[6] = 191.613
        initial_guess[7] = 60.652
        initial_guess[8] = 134.241
        initial_guess[9] = 43.548 #75.8
        initial_guess[10] = 122.268
        initial_guess[11] = 261.932
        initial_guess[12] = 81.774#82.244
        initial_guess[15] = 0.753
        initial_guess[16] = 135
        initial_guess[17] = 56.613
        initial_guess[18] = 60.484
        initial_guess[19] = 161.129
        initial_guess[20] = 101.613
        initial_guess[21] = 0
        initial_guess[22] = 180.761
        initial_guess[23] = 75
        initial_guess[24] = 135
        initial_guess[29] = 0.441
        initial_guess[30] = 83.226
        initial_guess[31] = 56.613
        initial_guess[32] = 64.839
        initial_guess[33] = 168.387
        initial_guess[34] = 101.613
        initial_guess[35] = 0
        initial_guess[36] = 168.387
        initial_guess[37] = 91.129
        initial_guess[38] = 126.774
    return initial_guess
