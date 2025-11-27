❯ python src/train_and_inference_bert_23rd_place.py
Using device: cuda
Computing input arrays...
Tokenizing: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6079/6079 [00:16<00:00, 363.11it/s]
Tokenizing: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6079/6079 [00:09<00:00, 608.40it/s]
Tokenizing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 476/476 [00:01<00:00, 434.01it/s]
Tokenizing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 476/476 [00:00<00:00, 828.72it/s]
Starting Fold 1/5
Fold 0: Training Question Only
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:45<00:00, 13.47it/s]
Epoch 1: Train Loss 0.1598478869142893, Valid Loss 0.15075088189424654
         Train Spearmanr 0.2256, Valid Spearmanr (avg) 0.2893, Valid Spearmanr (last) 0.2867
         elapsed: 50.69356179237366s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:47<00:00, 12.73it/s]
Epoch 2: Train Loss 0.14510006482075705, Valid Loss 0.14738859961691655
         Train Spearmanr 0.3273, Valid Spearmanr (avg) 0.2990, Valid Spearmanr (last) 0.3036
         elapsed: 53.813042402267456s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:48<00:00, 12.42it/s]
Epoch 3: Train Loss 0.13882942757520236, Valid Loss 0.14736217922089914
         Train Spearmanr 0.3695, Valid Spearmanr (avg) 0.3043, Valid Spearmanr (last) 0.3004
         elapsed: 54.994043588638306s
Fold 0: Training Q and A
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:48<00:00, 12.43it/s]
Epoch 1: Train Loss 0.1762506707669481, Valid Loss 0.16643633575815903
         Train Spearmanr 0.2802, Valid Spearmanr (avg) 0.4008, Valid Spearmanr (last) 0.3798
         elapsed: 54.9329571723938s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:48<00:00, 12.50it/s]
Epoch 2: Train Loss 0.16166838927586613, Valid Loss 0.16315777560597972
         Train Spearmanr 0.4220, Valid Spearmanr (avg) 0.4195, Valid Spearmanr (last) 0.4122
         elapsed: 54.67571496963501s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:48<00:00, 12.50it/s]
Epoch 3: Train Loss 0.15389580987884025, Valid Loss 0.16328562815722666
         Train Spearmanr 0.5064, Valid Spearmanr (avg) 0.4224, Valid Spearmanr (last) 0.4137
         elapsed: 54.66778802871704s
Starting Fold 2/5
Fold 1: Training Question Only
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:48<00:00, 12.43it/s]
Epoch 1: Train Loss 0.15986155761454843, Valid Loss 0.1500034696961704
         Train Spearmanr 0.2169, Valid Spearmanr (avg) nan, Valid Spearmanr (last) nan
         elapsed: 54.94041848182678s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:49<00:00, 12.41it/s]
Epoch 2: Train Loss 0.14494071533217243, Valid Loss 0.14767872924475292
         Train Spearmanr 0.3000, Valid Spearmanr (avg) nan, Valid Spearmanr (last) nan
         elapsed: 55.03963494300842s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:48<00:00, 12.43it/s]
Epoch 3: Train Loss 0.13852914751164222, Valid Loss 0.14682615342500963
         Train Spearmanr 0.3418, Valid Spearmanr (avg) nan, Valid Spearmanr (last) nan
         elapsed: 54.927711486816406s
Fold 1: Training Q and A
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.56it/s]
Epoch 1: Train Loss 0.17663668728384532, Valid Loss 0.16580001480485262
         Train Spearmanr 0.2755, Valid Spearmanr (avg) nan, Valid Spearmanr (last) nan
         elapsed: 50.316182136535645s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.56it/s]
Epoch 2: Train Loss 0.16166163097429825, Valid Loss 0.16455491593009547
         Train Spearmanr 0.4187, Valid Spearmanr (avg) nan, Valid Spearmanr (last) nan
         elapsed: 50.32912468910217s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.53it/s]
Epoch 3: Train Loss 0.15410852763115576, Valid Loss 0.1637196156539415
         Train Spearmanr 0.5011, Valid Spearmanr (avg) nan, Valid Spearmanr (last) nan
         elapsed: 50.3846001625061s
Starting Fold 3/5
Fold 2: Training Question Only
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.75it/s]
Epoch 1: Train Loss 0.15963147742379652, Valid Loss 0.14999635860716043
         Train Spearmanr 0.2190, Valid Spearmanr (avg) 0.3002, Valid Spearmanr (last) 0.2923
         elapsed: 49.795132875442505s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.73it/s]
Epoch 2: Train Loss 0.14525872032697262, Valid Loss 0.14766603178883853
         Train Spearmanr 0.3022, Valid Spearmanr (avg) 0.3029, Valid Spearmanr (last) 0.3050
         elapsed: 49.76871943473816s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.62it/s]
Epoch 3: Train Loss 0.13864607806317508, Valid Loss 0.1473303293123057
         Train Spearmanr 0.3500, Valid Spearmanr (avg) 0.3085, Valid Spearmanr (last) 0.3050
         elapsed: 50.11627006530762s
Fold 2: Training Q and A
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.68it/s]
Epoch 1: Train Loss 0.1768615867721995, Valid Loss 0.1673446823107569
         Train Spearmanr 0.2685, Valid Spearmanr (avg) 0.4050, Valid Spearmanr (last) 0.3788
         elapsed: 49.88224148750305s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.69it/s]
Epoch 2: Train Loss 0.16199392713851443, Valid Loss 0.16515759800217653
         Train Spearmanr 0.4118, Valid Spearmanr (avg) 0.4154, Valid Spearmanr (last) 0.4024
         elapsed: 50.04401612281799s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.75it/s]
Epoch 3: Train Loss 0.1541866566250591, Valid Loss 0.16501513260759806
         Train Spearmanr 0.4985, Valid Spearmanr (avg) 0.4190, Valid Spearmanr (last) 0.4093
         elapsed: 49.69462871551514s
Starting Fold 4/5
Fold 3: Training Question Only
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.72it/s]
Epoch 1: Train Loss 0.16009313020070917, Valid Loss 0.1497155899476064
         Train Spearmanr 0.2219, Valid Spearmanr (avg) 0.2916, Valid Spearmanr (last) 0.2906
         elapsed: 49.759793281555176s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.69it/s]
Epoch 2: Train Loss 0.14544816755101478, Valid Loss 0.14672499527468494
         Train Spearmanr 0.3214, Valid Spearmanr (avg) 0.3033, Valid Spearmanr (last) 0.3023
         elapsed: 50.04531502723694s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.65it/s]
Epoch 3: Train Loss 0.13896803032165686, Valid Loss 0.1461631303751155
         Train Spearmanr 0.3622, Valid Spearmanr (avg) 0.3083, Valid Spearmanr (last) 0.3068
         elapsed: 49.99753737449646s
Fold 3: Training Q and A
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.60it/s]
Epoch 1: Train Loss 0.17636515638840042, Valid Loss 0.16510213852712982
         Train Spearmanr 0.2793, Valid Spearmanr (avg) 0.4074, Valid Spearmanr (last) 0.3876
         elapsed: 50.15220069885254s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.64it/s]
Epoch 2: Train Loss 0.16172300020575917, Valid Loss 0.16257339715957642
         Train Spearmanr 0.4235, Valid Spearmanr (avg) 0.4206, Valid Spearmanr (last) 0.4055
         elapsed: 50.01955032348633s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.66it/s]
Epoch 3: Train Loss 0.15357725166617647, Valid Loss 0.16201478703633734
         Train Spearmanr 0.5094, Valid Spearmanr (avg) 0.4246, Valid Spearmanr (last) 0.4158
         elapsed: 49.97019910812378s
Starting Fold 5/5
Fold 4: Training Question Only
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.74it/s]
Epoch 1: Train Loss 0.1597834974527359, Valid Loss 0.15031848945900014
         Train Spearmanr 0.2326, Valid Spearmanr (avg) 0.2874, Valid Spearmanr (last) 0.2844
         elapsed: 49.69618368148804s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.73it/s]
Epoch 2: Train Loss 0.1453295431197866, Valid Loss 0.14645186262695412
         Train Spearmanr 0.3198, Valid Spearmanr (avg) 0.2979, Valid Spearmanr (last) 0.2981
         elapsed: 49.73447012901306s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.69it/s]
Epoch 3: Train Loss 0.13873474226382218, Valid Loss 0.14651724402057498
         Train Spearmanr 0.3641, Valid Spearmanr (avg) 0.3025, Valid Spearmanr (last) 0.3050
         elapsed: 49.841997385025024s
Fold 4: Training Q and A
Epoch 1/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.73it/s]
Epoch 1: Train Loss 0.17594045452087334, Valid Loss 0.16704782480864147
         Train Spearmanr 0.2781, Valid Spearmanr (avg) 0.4118, Valid Spearmanr (last) 0.3860
         elapsed: 49.741395711898804s
Epoch 2/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.65it/s]
Epoch 2: Train Loss 0.16119841393083334, Valid Loss 0.1650318969041109
         Train Spearmanr 0.4218, Valid Spearmanr (avg) 0.4171, Valid Spearmanr (last) 0.3992
         elapsed: 49.98895573616028s
Epoch 3/3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:44<00:00, 13.64it/s]
Epoch 3: Train Loss 0.15342496014445237, Valid Loss 0.16496197134256363
         Train Spearmanr 0.5082, Valid Spearmanr (avg) 0.4191, Valid Spearmanr (last) 0.4082
         elapsed: 50.036784172058105s
Training LightGBM Stacking...
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[64]    valid_0's l2: 0.0149884
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[75]    valid_0's l2: 0.0143589
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[76]    valid_0's l2: 0.0157309
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[50]    valid_0's l2: 0.0149108
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[85]    valid_0's l2: 0.0157189
Finished LightGBM for question_asker_intent_understanding
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[141]   valid_0's l2: 0.0246939
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[77]    valid_0's l2: 0.0264459
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[76]    valid_0's l2: 0.0275743
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[53]    valid_0's l2: 0.0280632
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[80]    valid_0's l2: 0.0258682
Finished LightGBM for question_body_critical
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[53]    valid_0's l2: 0.0219167
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[37]    valid_0's l2: 0.0250293
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[31]    valid_0's l2: 0.0213783
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[51]    valid_0's l2: 0.0212591
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[51]    valid_0's l2: 0.0218305
Finished LightGBM for question_conversational
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[34]    valid_0's l2: 0.115379
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[46]    valid_0's l2: 0.106594
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[33]    valid_0's l2: 0.106906
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[55]    valid_0's l2: 0.104589
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[56]    valid_0's l2: 0.11513
Finished LightGBM for question_expect_short_answer
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[68]    valid_0's l2: 0.0732152
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[44]    valid_0's l2: 0.0674381
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[59]    valid_0's l2: 0.0695825
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[152]   valid_0's l2: 0.0710699
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[71]    valid_0's l2: 0.0684181
Finished LightGBM for question_fact_seeking
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[79]    valid_0's l2: 0.0771977
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[72]    valid_0's l2: 0.071768
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[119]   valid_0's l2: 0.0800105
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[57]    valid_0's l2: 0.0755889
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[40]    valid_0's l2: 0.0859781
Finished LightGBM for question_has_commonly_accepted_answer
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[62]    valid_0's l2: 0.015692
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[73]    valid_0's l2: 0.015244
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[87]    valid_0's l2: 0.0158172
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[57]    valid_0's l2: 0.0165435
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[105]   valid_0's l2: 0.0153427
Finished LightGBM for question_interestingness_others
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[34]    valid_0's l2: 0.0259888
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[62]    valid_0's l2: 0.0237714
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[61]    valid_0's l2: 0.0260165
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[55]    valid_0's l2: 0.0237882
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[51]    valid_0's l2: 0.0253021
Finished LightGBM for question_interestingness_self
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[47]    valid_0's l2: 0.0660283
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[56]    valid_0's l2: 0.0751652
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[54]    valid_0's l2: 0.0675584
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[58]    valid_0's l2: 0.0652524
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[80]    valid_0's l2: 0.0704558
Finished LightGBM for question_multi_intent
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[14]    valid_0's l2: 0.000972518
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[32]    valid_0's l2: 0.00240504
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[13]    valid_0's l2: 0.00146544
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[25]    valid_0's l2: 0.00298774
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[12]    valid_0's l2: 0.00246688
Finished LightGBM for question_not_really_a_question
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[56]    valid_0's l2: 0.101063
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[44]    valid_0's l2: 0.0982688
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[55]    valid_0's l2: 0.0959532
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[71]    valid_0's l2: 0.0986519
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[46]    valid_0's l2: 0.101397
Finished LightGBM for question_opinion_seeking
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[61]    valid_0's l2: 0.0513077
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[68]    valid_0's l2: 0.0614246
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[59]    valid_0's l2: 0.0624514
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[55]    valid_0's l2: 0.0534408
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[76]    valid_0's l2: 0.0593787
Finished LightGBM for question_type_choice
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[80]    valid_0's l2: 0.0127389
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[39]    valid_0's l2: 0.0138257
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[70]    valid_0's l2: 0.0165459
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[63]    valid_0's l2: 0.012541
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[35]    valid_0's l2: 0.0117175
Finished LightGBM for question_type_compare
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[27]    valid_0's l2: 0.0055212
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[12]    valid_0's l2: 0.0039651
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[31]    valid_0's l2: 0.00432424
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[61]    valid_0's l2: 0.00525004
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[71]    valid_0's l2: 0.00454612
Finished LightGBM for question_type_consequence
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[21]    valid_0's l2: 0.00937476
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[41]    valid_0's l2: 0.00673478
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[48]    valid_0's l2: 0.0110965
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[37]    valid_0's l2: 0.0100308
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[63]    valid_0's l2: 0.0114378
Finished LightGBM for question_type_definition
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[43]    valid_0's l2: 0.0193977
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[114]   valid_0's l2: 0.0203624
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[56]    valid_0's l2: 0.0203766
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[85]    valid_0's l2: 0.0183456
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[42]    valid_0's l2: 0.021637
Finished LightGBM for question_type_entity
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[44]    valid_0's l2: 0.0648205
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[49]    valid_0's l2: 0.0572207
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[60]    valid_0's l2: 0.0612873
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[86]    valid_0's l2: 0.0544081
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[72]    valid_0's l2: 0.0658576
Finished LightGBM for question_type_instructions
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[67]    valid_0's l2: 0.0616027
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[34]    valid_0's l2: 0.057795
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[47]    valid_0's l2: 0.0599794
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[35]    valid_0's l2: 0.0556912
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[76]    valid_0's l2: 0.0567903
Finished LightGBM for question_type_procedure
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[106]   valid_0's l2: 0.0715428
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[49]    valid_0's l2: 0.078504
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[53]    valid_0's l2: 0.0742615
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[94]    valid_0's l2: 0.0718406
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[52]    valid_0's l2: 0.0747585
Finished LightGBM for question_type_reason_explanation
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[105]   valid_0's l2: 0.00105409
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[1]     valid_0's l2: 2.18513e-06
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[37]    valid_0's l2: 6.32142e-05
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[31]    valid_0's l2: 0.000462412
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[44]    valid_0's l2: 0.000204765
Finished LightGBM for question_type_spelling
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[42]    valid_0's l2: 0.0228281
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[51]    valid_0's l2: 0.0218295
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[46]    valid_0's l2: 0.0225555
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[51]    valid_0's l2: 0.0218574
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[87]    valid_0's l2: 0.0219494
Finished LightGBM for question_well_written
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[37]    valid_0's l2: 0.0123196
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[31]    valid_0's l2: 0.0117292
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[88]    valid_0's l2: 0.0119849
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[33]    valid_0's l2: 0.011831
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[88]    valid_0's l2: 0.012879
Finished LightGBM for answer_helpful
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[93]    valid_0's l2: 0.00919795
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[63]    valid_0's l2: 0.00948094
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[105]   valid_0's l2: 0.00958709
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[89]    valid_0's l2: 0.00932356
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[105]   valid_0's l2: 0.00913413
Finished LightGBM for answer_level_of_information
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[16]    valid_0's l2: 0.00701776
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[15]    valid_0's l2: 0.00660055
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[29]    valid_0's l2: 0.00759607
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[31]    valid_0's l2: 0.00768093
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[68]    valid_0's l2: 0.00770527
Finished LightGBM for answer_plausible
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[51]    valid_0's l2: 0.00472963
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[20]    valid_0's l2: 0.00473909
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[58]    valid_0's l2: 0.00528291
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[39]    valid_0's l2: 0.00470457
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[111]   valid_0's l2: 0.00639827
Finished LightGBM for answer_relevance
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[63]    valid_0's l2: 0.0142948
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[34]    valid_0's l2: 0.0147456
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[53]    valid_0's l2: 0.0147935
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[62]    valid_0's l2: 0.014786
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[58]    valid_0's l2: 0.0142936
Finished LightGBM for answer_satisfaction
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[45]    valid_0's l2: 0.0800196
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[56]    valid_0's l2: 0.0699507
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[79]    valid_0's l2: 0.0748547
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[92]    valid_0's l2: 0.0635762
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[49]    valid_0's l2: 0.07567
Finished LightGBM for answer_type_instructions
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[60]    valid_0's l2: 0.0508317
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[103]   valid_0's l2: 0.0484291
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[40]    valid_0's l2: 0.0450073
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[38]    valid_0's l2: 0.0412428
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[48]    valid_0's l2: 0.0504484
Finished LightGBM for answer_type_procedure
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[73]    valid_0's l2: 0.0869186
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[91]    valid_0's l2: 0.0856899
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[80]    valid_0's l2: 0.0885726
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[77]    valid_0's l2: 0.088592
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[112]   valid_0's l2: 0.0959114
Finished LightGBM for answer_type_reason_explanation
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[37]    valid_0's l2: 0.0094479
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[22]    valid_0's l2: 0.00908056
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[37]    valid_0's l2: 0.009921
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[28]    valid_0's l2: 0.00906674
Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[42]    valid_0's l2: 0.0104896
Finished LightGBM for answer_well_written
Submission saved to submission.csv