import statsmodels.api as sm
from scipy.stats import norm, kstest, shapiro, normaltest, anderson
import pandas as pd
from matplotlib import pyplot


class test_normal():
    def __init__(self, data=None, example_mode=None):
        if example_mode or data is None:
            self.data = pd.Series(data=norm.rvs(size=10_000))
        else:
            self.data = data

    def run(self):
        pass
        self.histogram_test()
        self.QQ_test()
        self.Kolmogorov_Smirnov_test()
        self.Shapiro_Wilk_test()
        self.normal_test()
        self.anderson_darling_test()

    def histogram_test(self):
        pyplot.hist(self.data)

    def QQ_test(self):
        sm.qqplot(self.data, line='45')
        pyplot.show()

    def Kolmogorov_Smirnov_test(self):
        ks_statistic, p_value = kstest(self.data, 'norm')
        print('KS statistic will be 0 if distribution is normal.')
        print('The P-Value is used to decide whether the difference is large enough to reject the null hypothesis:')
        print('\t- If the P-Value of the KS Test is larger than 0.05, we assume a normal distribution.')
        print('\t- If the P-Value of the KS Test is smaller than 0.05, we do not assume a normal distribution.')
        print(f'KS statistic: {ks_statistic:.4}, P-Value: {p_value:.4}')
        print('normal distribution') if ks_statistic<0.8 and p_value >0.05 else print('not normal distribution')

    def Shapiro_Wilk_test(self):
        print('running Shapiro-Wilk test')
        data = self.data.sample(5000) if self.data.shape[0] > 5000 else self.data
        stat, p = shapiro(data)
        print(f'stat: {stat:.4}, p: {p:.4}')
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')

    def normal_test(self):
        print('running D’Agostino’s K^2 test')
        stat, p = normaltest(self.data)
        print(f'stat: {stat:.4}, p: {p:.4}')
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')

    def anderson_darling_test(self):
        print('running Anderson-Darling test')
        result = anderson(self.data)
        print(f'stat: {result.statistic:.4}')
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


if __name__ == '__main__':
    n = test_normal()
    n.run()
