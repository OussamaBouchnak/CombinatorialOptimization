# Job Scheduling Optimization with Mold Constraints  

👤 **Author**: Oussama Bouchnak  

---

## 📋 Overview  
This project addresses the **job scheduling problem on two parallel machines** with **mold constraints**. Each job requires a specific mold, and a mold cannot be used simultaneously on multiple machines.  

---

## 🎯 Objective  
**Minimize the makespan** (total execution time) while respecting mold constraints to optimize resource utilization in a production environment.  

---

## 📊 Data  
Experiments were conducted on **2100 instances** categorized by:  
- **Number of jobs (n):** 50, 100, 150, 200, 250, 300, 400  
- **Number of molds (m):** 2, 5, 10, 15  

---

## 🔬 Methodology  

### 🔧 Implemented Algorithms  
1. **LPT** (Longest Processing Time) – Prioritize longest jobs first.  
2. **SPT** (Shortest Processing Time) – Prioritize shortest jobs first.  
3. **Tabu Search** – Metaheuristic with short-term memory.  
4. **Simulated Annealing** – Inspired by metallurgical cooling processes.  
5. **Genetic Algorithm** – Evolutionary approach with natural selection.  

---

## 📏 Evaluation Metrics  
- **Average Gap (%)**: Relative deviation from the lower bound.  
- **Optimal Solutions (%)**: Proportion of instances solved optimally.  
- **Execution Time**: Average time per instance.  

---

## 📊 Algorithm Performance  

### SPT vs LPT Comparison  
| Metric                | SPT         | LPT         |  
|-----------------------|-------------|-------------|  
| **Optimal Solutions** | 40%         | 61.9%       |  
| **Average Gap**       | 1.35%       | 1.32%       |  
| **Worst Case Gap**    | 36.92%      | 36.92%      |  

**Key Findings:**  
- LPT outperforms SPT across all classes (e.g., Class 1: 72% vs 50% optimality).  
- Prioritizing longer jobs (LPT) avoids mold conflicts more effectively.  

---

### Metaheuristics Comparison  
| Algorithm            | Optimal Solutions (%) | Avg Gap (%) | Strengths                          | Weaknesses                     |  
|----------------------|-----------------------|-------------|------------------------------------|--------------------------------|  
| **Tabu Search**      | 76.14%               | 1.30%       | High stability, avoids local traps | Complex implementation        |  
| **Genetic Algorithm**| 58.48%               | 1.36%       | Speed/accuracy balance             | Risk of stagnation            |  
| **Simulated Annealing**| 16.90%             | 1.74%       | Simple to implement                | Poor/random performance       |  

**Verdict:**  
**Tabu Search > Genetic Algorithm > Simulated Annealing**  

---

## 🚀 Conclusion  
- **LPT** is recommended for most scenarios due to its superior optimality rates.  
- **Tabu Search** dominates among metaheuristics, offering robust solutions for complex instances.  
- GPU acceleration significantly improves computation speed for large-scale problems.  

Explore the `src` directory for code implementation and `results` for detailed performance metrics.  

--- 

📝 **Note**: For execution steps or contributions, refer to the repository's documentation.  
