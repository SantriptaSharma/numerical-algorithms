# Programming Assignment: **Optimization Methods - Steepest Descent and Conjugate Gradient**

## **Objective**  
In this assignment, you will implement two optimization algorithms—**Steepest Descent (Gradient Descent)** and **Conjugate Gradient**—to minimize a given quadratic function. The goal is to understand the working principles of these methods and analyze their performance.

## **Problem Statement**  
You are tasked with minimizing a quadratic function of the form:  

$$
f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{A} \mathbf{x} - \mathbf{b}^T \mathbf{x},
$$

where:  
- $\mathbf{x}$ is the vector of variables ($n \times 1$),  
- $\mathbf{A}$ is a symmetric positive-definite matrix ($n \times n$),  
- $\mathbf{b}$ is a vector ($n \times 1$).  

The algorithms should be applied to find $\mathbf{x}^*$, which minimizes $f(\mathbf{x})$.  

---

## **Instructions**  

### 1. **Implement the following algorithms**:  
   - **Steepest Descent (Gradient Descent)**:  
     Update rule:  
     $$
     \mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k),
     $$
     where $\alpha_k$ is the step size (you can compute it analytically for quadratic functions).  

   - **Conjugate Gradient**:  
     Iterative method that builds a set of conjugate directions for faster convergence. Update rules involve calculating search directions $\mathbf{p}_k$ and step sizes $\alpha_k$ iteratively.  

### 2. **Input**:  
   - Define the following for testing your algorithms:  
     - $\mathbf{A} = \begin{bmatrix} 4 & 1 \\ 1 & 3 \end{bmatrix}$,  
     - $\mathbf{b} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$,  
     - Initial guess: $\mathbf{x}_0 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$.  

### 3. **Output**:  
   - For each method, print:  
     - The optimal solution $\mathbf{x}^*$,  
     - The value of $f(\mathbf{x}^*)$,  
     - The number of iterations required for convergence.  

### 4. **Visualization**:  
   - Plot the convergence of $f(\mathbf{x}_k)$ against the iteration number for both methods on the same graph.  
   - Add appropriate legends, titles, and axis labels.  

---

## **Submission Requirements**  

1. **Report**: Write a short report (1–2 pages) that includes:  
   - A description of both algorithms,  
   - A summary of your observations about their performance,  
   - Insights about the convergence speed and stability of each method.  

2. **Plots**: Include the convergence plot in your report.  

---

## **Bonus Task (Optional)**  
Modify the code to work with larger dimensions ($n > 2$) and test with a randomly generated symmetric positive-definite matrix $\mathbf{A}$. Discuss how the algorithms scale with problem size.  