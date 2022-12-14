import util_m
from DataMining import apriori
from DataMining import fp_growth
from DataMining.TheRomanceOfThreeKingdomsDataProcess import TheRomanceOfThreeKingdomsDataPreprocessing
from DataMining.theNationalCongressDataPreprocessing import TheNationalCongressDataPreprocessing
from sys import stdin
from DataMining.countWordFrequency import CountWordFrequency
from DataMining.insuranceDataProcessing import insurenceDataProcessing
from DataMining.patientDataProcessing import preprocessPatientDataHarshly, preprocessPatientDataMoreDetailedly
from DataMining.patternAssess import PatternAssess
import pandas as pd
from DataMining.shoppingBasketDataProcess import preprocessShoppingtData2

if __name__ == '__main__':
    # data = patientDataPreprocessing()
    ftype = 0
    support, confidence = 0.0, 0.0
    theNationalCongressDataPreprocessing = TheNationalCongressDataPreprocessing()
    threeKingdomsDataProcess = TheRomanceOfThreeKingdomsDataPreprocessing()
    cwf = CountWordFrequency()
    print("请输入\n"
          "选择频繁模式挖掘算法（1：Apriori  2：fp-growth）\n"
          "支持度 置信度(0,1)\n"
          "以空格分隔 如：“2 0.05 0.05 "
          "输入#退出")
    while True:
        print("请输入命令：")
        inputdata = stdin.readline().split(" ")
        # print(inputdata)
        if inputdata == ['#\n']:
            break
        if len(inputdata) < 3:
            continue
        ftype = inputdata[0]
        if ftype != "1" and ftype != "2":
            print("算法序号格式有误，请重试")
            continue
        try:
            support = float(inputdata[1])
            confidence = float(inputdata[2])
        except ValueError:
            support = 0
            confidence = 0
            print("输入有误,请重试")
            continue
        # data = preprocessPatientDataHarshly()
        # data = preprocessPatientDataMoreDetailedly()
        # data = insurenceDataProcessing()
        # data = threeKingdomsDataProcess.dataPreprocessing()
        data = theNationalCongressDataPreprocessing.dataPreprocessing()
        # data = preprocessShoppingtData2()
        # print('这里时预处理后的数据', data)
        if ftype == "1":
            apr = apriori.Apriori(datas=data, support=support, confidence=confidence)
            print(apr.freq_set)
            print(util_m.relate_rules2str(apr.rel_rules))
            pa = PatternAssess(data, apr.rel_rules)
            pa.calculateLiftAndCosine()
            pa.calculateAllConfidenced()
            pa.calculateKulczynski()
        else:
            # print(data)
            fp = fp_growth.FPGrowth(datas=data, support=support, confidence=confidence)
            # print("频繁模式树为：")
            # fp.fp_tree_root_point.print()
            print("频繁项集", fp.freq_set)
            # print("未格式化的关联规则",fp.relate_rule)
            print(util_m.relate_rules2str(fp.relate_rule))
            print("下面是模式评估相关量：**********************************************************************************")
            pa = PatternAssess(data, fp.relate_rule)
            pa.calculateLiftAndCosine()
            pa.calculateAllConfidenced()
            pa.calculateKulczynski()
            print()
        print("*************************")

'''
对于病人数据集结果不错的参数有 : 2 0.3 0.7 ;
购物篮1: 2 0.02 0.025
购物篮2 groceries: 

'''
