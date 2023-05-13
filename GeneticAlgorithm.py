import numpy as np
import random
import math
import concurrent.futures

# 遗传算法主体
class GeneticAlgorithm:
    def __init__(self, iter_times, loss_contribution, data_values, population_size, chromosome_length,
                 lower_bound, upper_bound, crossover_rate, mutation_rate, max_generations,
                 migration_interval, num_threads):
        # 适应度函数，用于计算种群中每个个体的适应度
        self.fitness_func = self.create_fitness_function(iter_times, loss_contribution, data_values)
        self.population_size = population_size #种群大小，即遗传算法中每一代中包含的个体数量。
        self.chromosome_length = chromosome_length #表示染色体的长度，也就是解向量的大小
        self.lower_bound = lower_bound  # 解的最小值
        self.upper_bound = upper_bound  # 解的最大值
        #交叉操作通常用于生成新的个体，并将父代个体的某些特征组合在一起，产生新的后代。
        self.crossover_rate = crossover_rate #交叉操作的概率
        #突变操作的概率。在遗传算法的每一代中，都有一定的概率对每个个体进行突变操作，以增加解的多样性突变操作的概率。
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations #遗传算法要进行的最大迭代次数
        #每隔多少个迭代周期进行一次迁移操作，即每进行self.migration_interval个迭代周期后，将适应度最高的个体发送给其他子群。
        self.migration_interval = migration_interval
        self.num_threads = num_threads #遗传算法中使用的线程数

        #初始化种群，二维数组，第一维度是种群中的个体数量，第二维度是每个个体的解向量，即染色体，长度为self.chromosome_length。
        self.population = self.initialize_population()

    '''
    iter_times:本轮参与方的迭代时间,数组
    loss_contribution：收敛贡献数组
    data_values:数据价值数组
    '''
    def create_fitness_function(self, iter_times, loss_contribution, data_values):
        #参数是个体的解向量
        def fitness_func(chromosome):
            data_values_total= 0.0
            loss_contribution_total=0.0
            free_time=0.0

            #本轮迭代的最长耗时
            each_client_time=np.array(iter_times)*np.array(chromosome)
            longest_time=np.max(each_client_time)

            # 使用enumerate()函数遍历数组
            for i, val in enumerate(chromosome):
                data_values_total+=val*data_values[i]
                loss_contribution_total+=val*loss_contribution[i]
                free_time+=longest_time-each_client_time[i]
            return data_values_total+math.exp(-loss_contribution_total)+math.exp(-free_time)
        return fitness_func


    '''
    在遗传算法中，种群（population）是指一个由多个个体（individual）组成的集合。
    每个个体对应着一组染色体（chromosome），它们是基因的载体，决定了个体的特征。
    初始化种群是指在种群中随机生成一些个体，即在给定的范围内随机初始化染色体，使其成为一组解向量。
    在初始化时，可以通过调整种群大小（population_size）来控制种群中个体的数量。
    initialize_population返回值为一个numpy数组，表示初始化后的种群。
    '''
    def initialize_population(self):
        # 在给定范围内随机初始化种群
        '''
        生成一个形状为 (self.population_size, self.chromosome_length) 的数组，
        数组中的每个元素都是在 self.lower_bound 和 self.upper_bound 之间
        '''
        return np.random.randint(self.lower_bound, self.upper_bound + 1, (self.population_size, self.chromosome_length))

    def fitness(self):
        # 计算种群中每个个体的适应度
        '''
        遍历种群中的每个个体，计算并获得了该个体的适应度值。np.array() 函数将计算出的适应度值构建成一个数组
        '''
        fitness_values = np.array([self.fitness_func(chromosome) for chromosome in self.population])
        return fitness_values

    # 使用轮盘赌选择法进行选择操作
    '''
    在遗传算法中，选择操作是根据个体的适应度值来选择一些个体作为下一代的父母。
    适应度值越高的个体被选中的概率就越大，从而增加其遗传到下一代的机会。
    适应度越高的个体被选中的概率越大。也就是说，选择操作后的种群中，适应度高的个体可能会有多个副本，而适应度低的个体可能会被完全排除
    比如说有一个四个个体的种群，适应度分别为[1, 2, 3, 4]。那么第一个个体被选中的概率就是1/(1+2+3+4)=0.1，
    第二个个体被选中的概率就是2/(1+2+3+4)=0.2
    这样做是为了让适应度高的个体有更多的机会进行交叉和变异，从而有更多的机会产生更优的解
    '''
    def selection(self):
        fitness_values = self.fitness() #计算出种群中每个个体的适应度值
        total_fitness = np.sum(fitness_values) #计算所有个体适应度值之和
        probabilities = fitness_values / total_fitness #计算出每个个体被选中的概率probabilities
        '''
        np.arange生成示从0到self.population_size-1的数组
        np.random.choice()从np.arange(self.population_size)这个长度为种群个体数量的数组中
        选取size=self.population_size`个元素，其选取概率由`p=probabilities`指定。这里的`probabilities`数组记录了每个个体被选中的概率，概率越大则被选中的概率就越高。例如，如果一个个体的适应度很高，那么其被选中的概率就比较大。选出的索引会存储在`selected_indices`数组中。最后，将原始种群`self.population`中对应的索引所表示的个体选中，组成新的种群，这样就完成了选择操作。
        '''
        selected_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probabilities)
        self.population = self.population[selected_indices]

    # 使用单点交叉进行交叉操作
    def crossover(self):
        '''
        range函数的第一个参数是起始位置，第二个参数是终止位置，第三个参数是步长。
        表示从0开始，每隔2个数取一个，一直取到self.population_size-1。
        其中i的取值就是0, 2, 4, ..., self.population_size-2。每次循环都会对i和i+1位置上的个体进行交叉操作。
        '''
        for i in range(0, self.population_size, 2):
            #random.random() 会随机生成一个0到1之间的实数，如果这个实数小交叉概率，则会进行交叉操作
            if random.random() < self.crossover_rate:
                '''
                随机选取单点交叉的交叉点位置，
                其中self.chromosome_length - 1表示染色体长度减去1，因为交叉点不能在第一位或最后一位，
                因为交叉点在第一位时，因为互换是从交叉点开始到结尾的基因互换（包括交叉点）
                random.randint函数会从这个区间内随机选取一个整数作为交叉点位置
                '''
                crossover_point = random.randint(1, self.chromosome_length - 1)
                '''
                单点交叉操作，将两个染色体从交叉点开始的位置进行交换。
                self.population[i, crossover_point:] 表示第 i 个个体从交叉点开始的位置到结尾的基因片段
                self.population[i + 1, crossover_point:] 表示第 i+1 个个体从交叉点开始的位置到结尾的基因片段。
                在交叉的过程中，这两个基因片段将进行交换。
                替换原有染色体是为了减小计算量
                '''
                self.population[i, crossover_point:], self.population[i + 1, crossover_point:] = self.population[i + 1, crossover_point:].copy(), self.population[i, crossover_point:].copy()

    # 使用高斯突变进行突变操作
    def mutation(self):
        for i in range(self.population_size):
            for j in range(self.chromosome_length):
                if random.random() < self.mutation_rate:
                    '''
                    给当前个体（染色体）的某个基因位置（j位置）添加一个服从正态分布的随机数，以达到变异的效果
                    np.random.normal(0, 1)生成的是一个以0为均值，1为标准差的正态分布随机数。
                    正态分布是对称的，所以生成的随机数有可能是正数也有可能是负数
                    选择正态分布的原因是因为它是最常见的分布之一，其形状为钟形，即大部分随机数会集中在均值附近（在这里是0），
                    远离均值的随机数出现的概率较低。这样的分布可以保证变异操作不会引入过大的变化，有利于算法的稳定性。
                    '''
                    self.population[i, j] += np.random.normal(0, 1)
                    # 保证突变后的解仍在给定范围内
                    #np.clip函数的作用是将数组中的元素限制在指定的范围内
                    #如果小于lower_bound，则将其替换为lower_bound；如果大于upper_bound，则将其替换为upper_bound
                    self.population[i, j] = np.clip(self.population[i, j], self.lower_bound, self.upper_bound)

    '''
    # 迁移操作，将每个子群中适应度最高的个体发送给其他子群
    def migrate(self):
        #np.argmax在适应度值列表中找到最大值，并返回其索引
        #axis=0参数表示沿着数组的第一个维度进行操作。二维数组，第一个维度通常是行，第二个维度是列。
        best_indices = np.argmax([self.fitness_func(chromosome) for chromosome in self.population], axis=0)
        for i in range(self.num_threads):
            if i != best_indices[i]:
                self.population[best_indices[i]] = self.population[best_indices[i], :].copy()
    '''

    def run(self):
        # 执行遗传算法的主循环
        for generation in range(self.max_generations):
            self.selection()
            self.crossover()
            self.mutation()

            '''
            if generation % self.migration_interval == 0:
                self.migrate()
            '''
        # 在所有的进化过程结束后，找出种群中适应度最小的个体
        fitness_values = self.fitness()
        best_index = np.argmax(fitness_values)
        self.best_solution = self.population[best_index]

        # 打印出最优解
        print("The best solution is: ", self.best_solution)


# 分布式遗传算法
'''
class DistributedGeneticAlgorithm:
    def __init__(self, num_threads, *args, **kwargs):
        self.num_threads = num_threads
        self.algorithms = [GeneticAlgorithm(*args, **kwargs) for _ in range(num_threads)]

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for algorithm in self.algorithms:
                executor.submit(algorithm.run)
'''
if __name__ == "__main__":
    iter_times=[9.23, 36.91, 10.82]
    loss_contributions=[68.2,147.5,33.1]
    data_values=[23.1,167.9,99.2]
    # 创建类的实例
    my_instance = GeneticAlgorithm(iter_times,loss_contributions,data_values,50,3,10,100,0.6,0.3,100,10,2)
    my_instance.run()