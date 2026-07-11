**Matrix Factorization for Modeling High-Dimensional Interactions in Class Ratemaking**

*Prepared by **Nana Kato, Suguru Fujita, Shunichi Nomura*

Presented to the 

33rd International Congress of Actuaries (ICA2026)

November 8th – 13th, 2026

*This paper has been prepared for the **33rd **International Congress of Actuaries.*

*The **Institute of **Actuaries** of** **Japa**n **(IAJ) and **the **International Actuarial Association (IAA) **assume no responsibility for opinions or content expressed in this paper or in the **accompanying** materials. Furthermore, the opinions and content contained herein do not reflect the views of the **IAJ or the IAA**.*

 Nana Kato, Suguru Fujita, and Shunichi Nomura

The IAJ and the IAA will ensure that all reproductions of the paper acknowledge the author(s) and include the above copyright statement.

## Abstract (accepted)

**Main Subject**: Non-Life Insurance
**Sub Subject**: Data Science/AI
**Keyword 01**: Matrix Factorization
**Keyword 02**: Class Rate-Making
**Keyword 03**: Interaction

This study introduces matrix factorization, a technique widely used in recommender systems, as a novel approach to insurance class rate-making. We address a critical challenge in actuarial science to handle the high dimensionality of categorical variables and their interactions, such as regions and car models. Although these variables often exhibit complex interactions, modern insurance portfolios typically suffer from sparse claim experience. While traditional statistical models have struggled to handle such high-dimensional interactions, our matrix factorization approach efficiently estimates them through a product of latent factors, which substantially reduces the number of parameters. Furthermore, our approach provides a simple and straightforward representation of the interactions, which is essential for accountability in setting pure premiums but is often overlooked in other machine learning methods. To demonstrate its performance, a comparative analysis was conducted using a real-world auto insurance dataset containing very high-dimensional variables. Our proposed model was implemented and optimized using cmfrec, a R/Python library for stable matrix factorization, with the number of latent factors and the regularization weight determined via cross-validation. The performance of our model was then benchmarked against a standard Generalized Linear Model (GLM) and a Generalized Linear Mixed Model (GLMM), both of which are common in actuarial practice. The results showed that our matrix factorization model not only accurately captured complex, non-linear interactions that traditional models struggle with but also effectively estimated rates for sparse segments where GLMs often fail to produce reliable results. This research validates that matrix factorization provides a powerful alternative for actuaries, offering a more precise and data-adaptive method for rate-making.


# Introduction

In the calculation of insurance premium rates, determining class rates often presents challenges when dealing with variables containing a large number of categories or missing data. For instance, in automobile insurance tariffs, variables with numerous categories—such as geographic region or vehicle manufacturer—are frequently employed, and interactions between these and other variables are often incorporated into the models.

   Interactions are crucial modeling elements for capturing effects that cannot be explained by individual variables alone. For example, a specific relationship may exist between a region and a vehicle manufacturer, where the loss ratio for a particular manufacturer is exceptionally high or low in a specific area. However, when considering interactions between variables that both have many categories, the number of possible combinations becomes enormous. This leads to segments with insufficient exposure due to data imbalance or segments with entirely missing data, further increasing the difficulty of modeling.

   While it is possible to account for these relationships by introducing interaction terms in traditional Generalized Linear Models (GLM), the accuracy of coefficient estimation declines and interpretability issues arise as the number of categories increases. Consequently, the introduction of alternative approaches is necessary for appropriate modeling.

One widely used alternative is the Generalized Linear Mixed Model (GLMM), which extends the Generalized Linear Model by introducing random effects for high-cardinality factors and their interactions (e.g. Hammad & Harby, 2016). Another important direction involves regularization methods applied to high-dimensional GLMs (e.g. Takahashi & Nomura, 2023). Both approaches shrink weak or unsupported interaction terms toward zero, reducing overfitting and improving stability. In parallel, machine learning methods such as neural networks (e.g. Richman & Wüthrich, 2024) have been widely adopted for their ability to automatically capture nonlinearities and interactions.

   On the other hand, in fields such as recommendation systems, Matrix Factorization is a widely used technique known for its high robustness against data sparsity and missing values. In addition, compared with machine learning methods, Matrix Factorization offers relatively high interpretability, as the latent factors can often be understood as underlying structures or relationships within the data. Furthermore, unlike random effects models or regularization methods, which tend to shrink unobserved interaction effects toward zero, Matrix Factorization can naturally impute such interactions through latent factor structures without forcing them to vanish. Although insurance data shares similar challenges regarding sparsity and missingness, there are very few instances where such matrix factorization methods have been applied to insurance ratemaking.

   In this study, we propose a new class ratemaking method utilizing matrix factorization. Specifically, we employ Matrix Factorization (MF) to explore an approach applicable even in environments with high-dimensional categorical data and missing values. To demonstrate the effectiveness of this method, we compare it with traditional class ratemaking techniques and discuss its advantages and challenges.

   The structure of this paper is as follows: Chapter 2 reviews traditional class ratemaking methods, focusing particularly on existing issues related to interactions. Chapter 3 provides an overview of matrix factorization techniques and discusses their application to insurance ratemaking. Chapter 4 presents the results of an analysis using empirical data to demonstrate the effectiveness of the proposed method. Finally, Chapter 5 offers concluding remarks and addresses directions for future research.

# Existing Methods for Class Ratemaking

## Generalized Linear Models (GLMs)

Generalized Linear Models (GLMs) have been widely utilized for calculating class rates. A GLM generalizes linear regression by allowing for flexible error structures and link functions. By relating the expected value of the response variable E[Y] to a linear combination of explanatory variables Xi via a link function g(∙), the model is expressed as follows:

g(EY  X1, X2, …, Xp])=* *β0+β1X1+β2X2+⋯+βpXp

Y: Response variable (e.g., claim frequency, claim severity, loss ratio)

- g(∙): Link function (e.g., identity, log, or logit function)

- β0, β1,⋯βp: Regression coefficients

-  X1,⋯Xp: Explanatory variables (e.g., age, gender, vehicle manufacturer, region, mileage) 

GLMs can incorporate interactions and non-linear relationships between explanatory variables. For instance, the combination of age and mileage may have an effect that differs from considering each factor in isolation. In such cases, the relationship can be captured by adding an interaction term XiXj as an explanatory variable:

g(EY  X1, X2, …, Xp])=* *β0+β1X1+β2X2+⋯+βpXp+i=1p-1j=i+1pβijXiXj

Here, βij represents the regression coefficient for the interaction between Xi and Xj. Additionally, if the response variable increases non-linearly with an explanatory variable, higher-order terms such as a quadratic term Xi2 can be added.

   However, when variables have many categories, these interaction terms significantly increase model complexity, leading to issues with interpretability and stability. For example, consider a categorical variable like "vehicle manufacturer" with hundreds or thousands of categories. If we consider the interaction between this and another categorical variable (e.g., region), the number of possible combinations becomes enormous. Explicitly including all such interaction terms in a GLM leads to the following problems:

Model Complexity: The sheer number of explanatory variables makes the model extremely complex, hindering interpretability and increasing computational costs.

Overfitting: Adding numerous interaction terms increases the risk that the model will overfit the training data, thereby reducing its predictive performance on unseen data.

Multicollinearity: When multiple variables are interrelated, multicollinearity can occur, making coefficient estimates unstable and difficult to interpret.

Sparsity: When data for specific category combinations are scarce or entirely missing, it becomes difficult to accurately estimate the coefficients for those interaction terms. Handling missing values requires careful consideration, as simple imputation methods may introduce bias.

   Interactions are crucial modeling elements for capturing effects that cannot be explained by individual variables alone. For example, a specific relationship may exist between a region and a vehicle manufacturer, where the loss ratio for a particular manufacturer is exceptionally high or low in a specific area. However, when considering interactions between variables that both have many categories, the number of possible combinations becomes enormous. This leads to segments with insufficient exposure due to data imbalance or segments with entirely missing data, further increasing the difficulty of modeling.

## Generalized Linear Mixed Models (GLMMs)

To address the limitations of GLMs—particularly the issues of over-parameterization and overfitting when dealing with high-dimensional categorical variables—the Generalized Linear Mixed Models (GLMMs) are often considered. GLMMs extend the standard GLMs by introducing random effects for high-cardinality factors and their interactions as well as the traditional fixed effects. 

Mathematically, by incorporating random effects corresponding to specific category levels, the model can be expressed by listing the individual variables as follows:

g(EY  X1, X2, …, Xp,u1,u2, …,uq])=* *β0+β1X1+β2X2+⋯+βpXp+u1Z1+u2Z2+⋯+uqZq

where u1,u2, …,uq represent the random effects assumed to follow a common distribution N(0,  δ2), and Z1,Z2, …,Zq are the dummy variables corresponding to the random effects. In the context of class ratemaking, fixed effects β1X1,β2X2,…,βpXp are typically used for standard explanatory variables, while random effects u1Z1,u2Z2,…,uqZq are introduced for high-cardinality factors and their interactions, such as vehicle manufacturer or region.

The main advantage of introducing random effects lies in their inherent partial pooling mechanism. When data for specific category combinations are scarce or missing, GLMMs assume that the random effects for these combinations are drawn from a common distribution. This distributional assumption naturally shrinks weak or unsupported interaction terms toward zero, reducing overfitting and improving stability.

However, while GLMMs provide a robust statistical framework for handling sparsity through regularization, they assume that unobserved or weakly supported interaction effects trend toward zero. They do not attempt to uncover underlying latent structures that might explain the missing data. Matrix Factorization, which will be introduced in Chapter 3, addresses this specific limitation by naturally imputing such interactions through latent factor structures without forcing them to vanish.

# Class Ratemaking Using Matrix Factorization

## Matrix Factorization

Matrix factorization (MF) is a technique originally developed in the fields of recommender systems and natural language processing. By decomposing high-dimensional data into low-dimensional latent factors, MF can efficiently capture the underlying structure of the data.

   In this section, we describe the problem setting and estimation methods of MF within the context of recommender systems. In a basic setup, the preferences of m users for n items are observed as a non-negative matrix X of dimensions m×n . However, in practice, only a small fraction of the elements in the large-scale matrix X are observed. Thus, X is typically a sparse matrix where most entries are missing. The objective of a recommender system is to learn the structure of X and appropriately impute the missing values. This allows the system to recommend suitable items by predicting user preferences for all items. Below, we introduce two widely used techniques: Singular Value Decomposition (SVD) and Non-negative Matrix Factorization (NMF).

### Singular Value Decomposition (SVD)

SVD is a method that decomposes an observed matrix X as follows:

X≈UΣVT

In this equation, U is an orthogonal matrix of left singular vectors, Σ is a diagonal matrix containing singular values, and V is an orthogonal matrix of right singular vectors. In recommender systems, we limit the rank of Σ (the number of non-zero diagonal elements) to a value much smaller than m or n. This allows us to map the observed matrix X into a low-dimensional latent factor space to learn user preferences.

   While standard algorithms exist to compute SVD efficiently for complete matrices, they cannot be directly applied to the sparse matrices found in recommender systems. Instead, parameters are estimated using gradient-based optimization or Alternating Least Squares (ALS). A key advantage of SVD is its ability to reduce dimensionality while filtering out noise. However, when the observed matrix X contains only non-negative values, SVD-based imputation may produce negative values. To avoid this, the following Non-negative Matrix Factorization is often more effective.

### Non-negative Matrix Factorization (NMF)

NMF is a technique that factorizes a non-negative observation matrix X as follows:

X≈ABT

   In this formulation, A and B are non-negative matrices with a small number of columns, representing the latent factors of users and items, respectively. Within the context of recommendation systems, A represents user features while B represents item features; their product is utilized to predict user preferences. A key characteristic of NMF is that the latent factor matrices A and B are constrained to non-negative values and often contain many zeros. This distinct contrast between zero and non-zero elements enhance the interpretability of the estimation results.

   The non-negative matrices A and B are typically estimated using gradient-based optimization methods or Alternating Least Squares (ALS). However, since these methods are prone to converging to local optima, providing appropriate initial values is essential. Alternatively, estimation can be stabilized using regularization, as discussed in the following section.

## Proposed Matrix Factorization Model

Research on the application of matrix factorization to premium rating has also been advancing in international studies. For non-life instance, Seo et al. (2022) used sparse non-negative matrix factorization to extract the relationship between aggressive driving behavior and driving risk as interpretable low-rank latent risk factors, and successfully distinguished between high-risk and low-risk driving behaviors. Xie and Gan (2022) and Xie et al. (2024) applied sparse non-negative matrix factorization with fuzzy clustering to auto insurance claim data for the assessment of relative territory risk.  These studies demonstrate that matrix factorization can provide models that are more flexible and offer higher interpretability compared to traditional statistical methods, and it is anticipated as a novel approach in the field of actuarial science. 

   In this study, we propose an approach that directly applies matrix factorization to ratemaking. We utilize the following model, which further refines the non-negative matrix factorization model introduced in the previous section:

X≈ABT+μ1m1nT+bA1nT+1mbBT

Where A is an m×k matrix representing user latent factors, B is an n×k matrix representing item latent factors, μ is the global mean, and 1m, 1n are m and n dimensional vectors of ones. Furthermore, bA is an m dimensional vector representing user-specific biases, and bB is an n dimensional vector representing item-specific biases. The variables to be estimated in the above equation are the elements of A, B, bA, bB, and μ. These are estimated using the Alternating Least Squares (ALS) method under non-negativity constraints, with the addition of L2 regularization terms (penalty terms based on the sum of squares of each element). The optimization problem for parameter inference is formulated by

minimizeA,B,bA,bB,μ  X-ABT+μ1m1nT+bA1nT+1mbBTF2+λAF2+BF2

s.t.  elements of A, B are all non-negative,

where ∙F2 is the squared Frobenius norm, that is, the square sum of the observed elements.  Additionally, the number of columns k in the latent factor matrices A and B, which determines the rank of the interaction term ABT, is selected along with the weight λ of the L2 regularization terms from candidate values through cross-validation.

   As the objective of this study is the application to class ratemaking, the observation matrix X consists of claim costs, specifically the historical pure premium rates. In recommendation systems, the rows and columns of the observation matrix X typically represent users and items, respectively; however, in this study's premium rating application, we correspond two risk factors with numerous categories to the rows and columns. Risk factors with many categories are assumed to include variables such as geographic regions or vehicle models, as discussed in the application examples in the next section. 

   As mentioned in Chapter 2, estimating interactions between risk factors with many categories poses a challenge due to missing data in specific category combinations. However, in our proposed approach, interactions are represented by a low-rank matrix using the product of latent factor matrices. Consequently, even for combinations with missing data, appropriate predictions that account for the effects of the factors can be expected.

# Applications

## Overview of the Analysis

In this section, we estimate pure premium rates by vehicle model and geographic region using automobile insurance claims data, specifically employing matrix factorization techniques. To provide a baseline for evaluation, we first present estimation results from two conventional methods: a Generalized Linear Model (GLM) without interaction terms and a Generalized Linear Mixed Model (GLMM) that treats interactions as random effects. Following this, we introduce the estimation results obtained from our proposed matrix factorization approach as a comparative counterpart. Based on these results, we discuss the practical effectiveness and advantages of applying matrix factorization to the determination of class-based premium rates in the insurance industry.

## Dataset

For this analysis, we utilize the brvehins1 dataset from the CASdatasets library, which comprises Brazilian automobile insurance data. These data were originally sourced and processed from the AUTOSEG automobile insurance statistical system. The dataset consists of 1,965,355 records, containing detailed information on exposures, premiums, and claim amounts. Available attributes include gender, age, vehicle model and group, vehicle year, region, and state.

   In this study, we focus specifically on risk factors with a high number of categories: vehicle model and geographic region. The target variable for prediction is the historical pure premium rate, defined as the claim cost (total claim amount divided by total exposure).

   Figure 4.2.1 displays the actual claim costs (historical pure premium rates) for the selected Honda vehicle models across various geographic regions. Given that the full dataset contains 4,259 vehicle model categories and 40 regional categories—making direct visualization difficult—we have refined the scope for our results display. Specifically, we focus on Honda vehicle models, further narrowing the selection to 48 representative categories where the total exposure is 10 or greater. Furthermore, since it is statistically challenging to determine pure premium rates from historical data when the volume of contracts is low, we only utilize data from cells (vehicle model × region combinations) with a total exposure of 100 or more. All other combinations are treated as missing values to be estimated via the proposed matrix factorization approach.

![fig_4_2_1](fig_4_2_1.png)

Figure 4.2.1: Actual Claim Costs by Vehicle Model and Region

## GLM without Interaction Terms

In actuarial practice, explicitly defining interaction terms for a vast number of category combinations is often considered technically challenging and computationally complex. Consequently, Generalized Linear Models (GLM) without interaction terms are commonly used as a pragmatic baseline. We first apply this fundamental approach to our dataset.

   As previously described, the model is defined as follows:

Let Yij be the total claim amount for vehicle model i and area j, and Eij be the corresponding exposure. The model is defined as follows:

Yij~Poisson(λijEij )

lnEYij ~lnEij+ β0+ αi+ τj

Where:

EYij is the expected total claim cost.

lnEij serves as the offset term to account for varying exposure levels.

β0 is the intercept.

αi and τj represent the main effects of the vehicle model and geographic area, respectively. Note that this specification assumes no interaction between the vehicle model and the area. The model is fitted using only the observed cells where data is present.

   The estimation results for pure premium rates by vehicle model and region are shown in Figure 4.3.1. and Figure 4.3.2.. In Figure 4.3.2.., consistent with the previous section, the uncolored (white) areas represent missing values where data was insufficient for estimation.

![fig_4_3_1](fig_4_3_1.png)

Figure 4.3.1: Heatmap of Predicted Pure Premium Rates using a Main-Effects GLM

(GLM without Interaction Terms)

![fig_4_3_2](fig_4_3_2.png)

Figure 4.3.2: Estimated Pure Premium Rates by Vehicle Model and Region

the uncolored (white) areas represent **missing values** where data was insufficient for estimation (GLM without Interaction Terms)

The characteristics of the estimation results are summarized below:

		- Simplicity and Limitations in Interaction Modeling: While this approach is straightforward and easy to interpret, it fails to account for interaction effects. Consequently, the heatmap exhibits proportional color transitions across vehicle models and regions. Furthermore, the estimated values for non-missing cells deviate from the historical data, indicating that the model does not necessarily align with actual risk profiles.

		- Extrapolation to Missing Values: By applying uniform coefficients across all categories, it is possible to calculate rates for cells with missing data. However, for categories that were entirely absent from the model-building dataset (the white spaces in the heatmap), the results represent a simple extrapolation rather than an estimation based on observed data.

## GLMM with Interactions as Random Effects

Generalized Linear Mixed Model (GLMM) is an extension of the GLM that allows for the inclusion of both fixed effects and random effects, making it particularly useful for modeling correlated or clustered data. In this section, we consider a model that accounts for interaction effects by treating the variables from the main-effects GLM (Section 4.3) as fixed effects, while incorporating the interaction between vehicle model and geographic area as a random effect.

The model is specified as follows:

Yij~Poisson(λijEij )

lnEYij ~lnEij+ β0+ αi+ τj+ zij

Where zij~N(0,  δ2) represents the random effect for the interaction between vehicle model i and area j. The definitions of the other variables in the equation are the same as those described in Section 4.3.

   The estimation results for the observed (non-missing) cells are shown in Figure 4.4.1 and 4.4.2. 

![fig_4_4_1](fig_4_4_1.png)

Figure 4.4.1: Estimated Pure Premium Rates by Vehicle Model and Region (GLMM with Random Effects)

(Note: Extrapolation results for all categories, including missing values, are omitted as the model cannot uniquely determine random effects for unobserved combinations.)

![fig_4_4_2](fig_4_4_2.png)

Figure 4.4.2: Estimated Pure Premium Rates by Vehicle Model and Region the uncolored (white) areas represent missing values where data was insufficient for estimation (GLMM with Random Effects)

The characteristics of the estimation results are summarized below:

		- Interaction Modeling for Observed Data: For non-missing cells, the GLMM successfully incorporates interaction effects, allowing for a more nuanced estimation than the main-effects GLM by capturing specific local variations.

		- Limitations Regarding Missing Values: A primary challenge of this approach is that the random effect zij is only estimable for observed combinations. For cells with zero exposure (missing data), the model lacks an empirical basis to predict the interaction, causing the estimate to revert to the main effects.

		- Violation of Distributional Assumptions: The fundamental assumption that interaction effects across all regions and models follow a single normal distribution may be too restrictive. Real-world insurance risks often exhibit complex local clusters that a simple N(0,  δ2) assumption fails to capture accurately.

		- Overall Effectiveness: Consequently, the GLMM approach is not particularly effective for datasets characterized by high sparsity, as it fails to provide reliable predictive power for the numerous missing category combinations where interaction effects are most needed.

## Matrix Factorization (MF)

In this section, we apply the proposed matrix factorization approach using the cmfrec library in R. This library is a standard tool for matrix factorization available in both R and Python. It implements various optimization algorithms, including gradient-based methods and Alternating Least Squares (ALS), and supports both L1 and L2 regularization. Additionally, the library incorporates specific initialization strategies designed by the author to avoid convergence to poor local optima.The model is formulated in 3.2.

   In terms of the model implementation, we utilize the Alternating Least Squares (ALS) algorithm for optimization alongside* *L2 regularization, both of which serve as the default settings within the library. To ensure that the estimated factors remain within a valid range for premium rating, the nonneg parameter is set to TRUE to enforce strict non-negativity constraints. Furthermore, the center parameter is set to FALSE to bypass mean-centering, thereby maintaining the original scale and integrity of the non-negative observation matrix.

- Hyperparameter Optimization via Cross-Validation

The number of latent factors k and the regularization weight λ were optimized through a 4-fold cross-validation using a grid search method. We selected the parameter combination that minimized the average Root Mean Square Error (RMSE), resulting in an optimal configuration of k=22 and λ=30.

- Model Training and Validation

To evaluate the predictive performance, we conducted a hold-out validation where 25% of the data was reserved as a test set, and the remaining 75% was used for model construction. The goodness of fit was assessed using the RMSE calculated against the test data. Figure 4.5.1 presents the Predicted vs. True Values of Pure Premium Rates (Matrix Factorization), where the correlation between the actual pure premiums and the model’s predictions is visualized. The concentration of data points along the identity line indicates that the matrix factorization model successfully captures the underlying risk patterns even for the hold-out test set.

- Estimation of Pure Premium Rates for All Categories

For the final estimation, the model was retrained using the entire dataset as input. This allowed us to estimate pure premium rates for all vehicle model and region combinations, including the cells originally treated as missing due to low exposure. The resulting heatmap for all categories is shown in Figure 4.5.2.

![fig_4_5_1](fig_4_5_1.png)

Figure 4.5.1: Predicted vs. True Values of Pure Premium Rates 

(Matrix Factorization)

![fig_4_5_2](fig_4_5_2.png)

Figure 4.5.2: Estimated Pure Premium Rates by Vehicle Model and Region (Matrix Factorization)

The characteristics of the estimation results are summarized below:

- Robust Estimation for Missing Categories: Unlike the GLMM, which struggles with unobserved cells, the matrix factorization approach successfully estimates pure premium rates for the entire matrix. For non-missing cells, the estimates remain highly consistent with the historical data presented in Section 4.2, while the missing cells are imputed with reasonable values based on the underlying latent factors of vehicle models and regions.

- Capture of Non-linear Interactions: The resulting heatmap (Figure 4) displays distinct non-linear patterns rather than the uniform, proportional changes seen in the main-effects GLM. This indicates that the model effectively captures complex interactions between specific vehicle models and geographic areas through the inner product of their respective latent factor matrices.

- Predictive Reliability: The high degree of correlation between predicted and true values shown in Figure 4.5.1 confirms that the model does not merely overfit the training data. Instead, it maintains strong predictive performance on hold-out samples, justifying its use as a reliable tool for practical premium rate setting.

By representing the interaction matrix as a low-rank product, the model filters out random noise inherent in sparse insurance data. This allows the system to identify "latent clusters" of risk—such as vehicle types with similar safety profiles or regions with comparable theft rates—leading to more stable and generalized predictions than traditional categorical models.

# Conclusion and Future Work

In this study, we focused on class-based premium rating and proposed a novel approach utilizing Matrix Factorization (MF). Through the application of this method to real-world insurance data, we verified that matrix factorization is highly effective for datasets characterized by a large number of categories and significant sparsity. Specifically, the latent factor model—originally developed for recommendation systems—demonstrated its capability to capture complex, non-linear interactions that traditional statistical models often fail to identify. Our findings suggest that MF provides a more robust and granular alternative to standard actuarial methods in the presence of high-dimensional risk factors.

  While this research confirms the effectiveness of the proposed approach, several avenues for further development remain:

- Integration of Side Information: One promising extension is the inclusion of "side information," such as demographic attributes or geographic characteristics. Preliminary experiments utilizing Collective Matrix Factorization (CMF)—which jointly factorizes the primary claim matrix alongside auxiliary attribute matrices for vehicle groups and population density—indicated potential improvements in predictive stability and accuracy. This approach allows the model to leverage shared characteristics across similar categories, effectively mitigating the challenges posed by extreme data sparsity in specific risk cells.

- Validation Across Diverse Datasets: This study utilized Brazilian automobile insurance data. To ensure the generalizability of the findings, it is essential to validate the model using datasets from other regions and different lines of business, such as homeowners or health insurance, where category sparsity is also a common challenge.

- Comparison with Deep Learning Approaches: Future research could also explore the trade-offs between Matrix Factorization and more complex deep learning architectures, such as Neural Collaborative Filtering (NCF), particularly regarding the balance between predictive power and the interpretability required in a regulated actuarial environment.

# References

**Dutang, C., & Charpentier, A. (2020).** *CASdatasets: A Collection of Actuarial Datasets*. R package version 1.0-11.

**Hammad, M. S., & Harby, G. A. (2016).** Using Multilevel Modeling for Group Health Insurance Ratemaking. *Predictive Modeling Applications in Actuarial Science: Volume 2, Case Studies in Insurance,* 126.

**Norberg, R. (1993).** Prediction of individual lifetime losses in disease insurance. *ASTIN Bulletin*, 23(1), 83–95.

**Richman, R., & Wüthrich, M. V. (2024).** High-cardinality categorical covariates in network regressions. *Japanese Journal of Statistics and Data Science*, 7(2), 921–965.

**Seo, H., Shin, J., Kim, K. H., Lim, C., & Bae, J. (2022). **Driving risk assessment using non-negative matrix factorization with driving behavior records. *IEEE Transactions on Intelligent Transportation Systems*, *23*(11), 20398–20412.** **

**Takahashi, A., & Nomura, S. (2023).** Automatic segmentation of insurance rating classes under ordinal constraints via group fused lasso. A*sia-Pacific Journal of Risk and Insurance*, 17(1), 113–142.

**Xie, S., & Gan, C. (2022).** Fuzzy clustering and non-negative sparse matrix approximation on estimating territory risk relativities. In *2022 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)* (pp. 1–8). IEEE.

**Xie, S., Gan, C., & Lawniczak, A. T. (2025).** Non-negative Sparse Matrix Factorization for Soft Clustering of Territory Risk Analysis. *Annals of Data Science*, *12*(1), 307–340.

	

2

**The Institute of Actuaries of Japan**

2F, Office Tower X, Harumi Island Triton Square, 1-8-10 Harumi, Chuo-ku, Tokyo, JAPAN

104-6002

			**e** secretariat@actuaries.jp  **w** www.actuaries.jp