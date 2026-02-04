# **XSSGAI: The First Ever AI-Powered XSS Payload Generator**

**XSSGAI** is the **first-ever AI-powered XSS (Cross-Site Scripting) payload generator**. Leveraging machine learning and deep learning techniques, it creates novel payloads based on patterns learned from a dataset of real-world XSS attacks. This groundbreaking tool is designed to assist security researchers and ethical hackers in identifying and mitigating potential XSS vulnerabilities by generating diverse and sophisticated attack vectors.

**XSSGAI v2.0** is the evolution of the first-ever AI-powered XSS payload generator. Moving away from traditional RNNs, this version leverages a **Transformer Architecture** the same technology behind modern LLMs, to synthesize sophisticated, context-aware Cross-Site Scripting attack vectors.

By utilizing **Multi-Axis Proportional Balancing** and **Nucleus (Top-P) Sampling**, v2.0 doesn't just predict characters; it understands the structural logic of bypasses, encoding layers, and tag-event relationships.

---

## **Features**

- **Transformer Brain**: Shifted from GRU/LSTM to a **6-Layer Transformer Encoder** with **Pre-Layer Normalization** (Norm-First) for superior training stability and long-range pattern recognition.

- **Nucleus (Top-P) Sampling**: Replaced rigid "Greedy" generation with probabilistic sampling. This allows the model to be "creative," exploring rare bypass syntaxes instead of repeating common patterns.

- **Multi-Axis Balancing**: A specialized data engine that ensures the model is equally proficient in common tags (like `<img>`) and rare, high-value tags (like `<math>` or `<details>`).

- **Contextual Encoding Augmentation**: Integrated support for **URL, Hex, and Double-URL encoding** within the neural logic, allowing the model to generate payloads that are pre-obfuscated for WAF bypass.

- **Set-Based Novelty Detection**: A high-performance  verification system that instantly tells you if a generated payload is a "Memorized" classic or a "Novel" AI original.

- **Heuristic Logic Layer**: A post-generation "Correction" system that ensures HTML symmetry (closing tags and quotes) while maintaining the neural-generated core.

- **Google Colab Support**: Easily run the tool in Google Colab for cloud-based execution without local setup, making it accessible to users without high-end hardware.

- **Open Source**: Fully open-source under the MIT License, encouraging community contributions and improvements.

---

## **Technical Specifications**

| Feature | Specification |
| --- | --- |
| **Model Type** | Transformer Encoder (Generative) |
| **Depth** | 6 Encoder Layers |
| **Sequence Length** | 384 Tokens (Supports deep obfuscation) |
| **Optimizer** | AdamW with StepLR Scheduler |
| **Loss Logic** | Cross-Entropy with 0.1 Label Smoothing |
| **Framework** | PyTorch / IPyWidgets |

---

## **Prerequisites**

To use this tool, ensure you have the following installed:

* **Python 3.10+**
* **PyTorch 2.x+**
* **CUDA-Enabled GPU** (Tesla T4 or better recommended)
* **IPyWidgets** (for the interactive GUI)

---

## **Installation**

### **Option 1: Run Locally**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AnonKryptiQuz/XSSGAI.git
   cd XSSGAI
   ```

2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure `requirements.txt` contains:
   
   ```plaintext
    torch>=2.0.0
    torchvision
    torchaudio
    numpy>=1.23.0
    pandas>=1.5.0
    scikit-learn
    matplotlib
    seaborn
    tqdm
    requests
    urllib3
    ipywidgets>=8.0.0
    IPython
    notebook
    jupyterlab
   ```

3. **Run the Jupyter Notebook:**

   ```bash
   jupyter notebook XSSGAI_AnonKryptiQuz.ipynb
   ```

   - You can also view previous versions of the notebook and other files inside the **[PREVIOUS\_VERSIONS](./PREVIOUS_VERSIONS)** folder.

4. **Run the Notebook**:
   - If you’re viewing the notebook for reference, all outputs (e.g., graphs, results) are already included.
   - If you want to regenerate payloads or experiment with the model, execute the cells sequentially.

---

### **Option 2: Run in Google Colab**

1. **Open Google Colab**:  
   Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

2. **Upload the Notebook and Dataset Files**:  
   - In Google Colab, click on the **folder icon** in the left sidebar to open the file browser.
   - Upload the following files:
     - The Jupyter Notebook file: `XSSGAI_AnonKryptiQuz.ipynb`.
     - The dataset files: Ensure the necessary files, such as `train_payloads.csv` and `test_payloads.csv`, are uploaded in the correct structure and accessible in the Colab environment.
    * *Optional:* You can also upload the `xss_transformer_v2.pth` file to save time and skip the training process if you'd like to continue testing from a pre-trained model.

3. **Run the Notebook**:  
   - Execute all the cells sequentially by selecting **Runtime > Run All** or by running each cell individually.
   - The notebook will automatically handle dataset processing, install necessary dependencies, train the model, and generate the payloads.

4. **View Results**:  
   - After running the cells, the generated payloads and any associated outputs (e.g., graphs, metrics) will be displayed directly in the notebook.
   - You can save the results or export them as needed.

---

## **Usage**

### **1. Set Parameters**

Adjust parameters in the notebook to optimize model performance:

* **MAX_LEN**: Set to **384** (supports complex obfuscation).
* **learning_rate**: Set to **0.0005**.
* **EPOCHS**: Set to **10**.
* **Temperature**: Adjust between **0.1** (stable) and **1.2** (creative).
* **Top-P**: Set to **0.9** (Nucleus Sampling).

### **2. Rapid Deployment (Skip Training)**

To save time (~60 mins), you can use pre-trained assets. Upload the following files from the GitHub to your working directory to skip the training phase:

* `xss_transformer_v2.pth` (The Neural Brain)
* `vocab.json` (Token Dictionary)
* `loss_history.json` & `transformer_loss.png`

### **3. Interactive GUI Features**

The v2.0 dashboard allows you to control the "DNA" of the generated payload:

* **Action**: Define JS intent (e.g., `ALERT`, `COOKIE`, `EVAL`).
* **Tag/Event**: Force specific injection points.
* **Entropy**: Control the randomness of the output.
* **Seed Text**: Provide breakout characters (e.g., `"><`) to guide the synthesis.

### **4. Experiment & Evaluate** 

Feel free to adjust these or other parameters and observe how changes impact model accuracy, payload generation, and overall performance. This flexibility allows you to fine-tune the generator for optimal results.

### **5. Viewing Results Without Downloading Files** 

If you prefer to view the results without downloading any files, you can simply view the generated PDF file: [XSSGAI_AnonKryptiQuz.pdf](./XSSGAI_AnonKryptiQuz.pdf), which contains all relevant output.

---

## **Example Synthesis**

```plaintext
[INSTRUCT] ACT:ALERT TAG:IMG EV:ONERROR STY:PLAIN [PAYLOAD]
>> <img src=x onerror=alert(1)> [MEMORIZED]

[INSTRUCT] ACT:COOKIE TAG:SVG EV:ONLOAD STY:ENCODED [PAYLOAD]
>> <svg/onload=%61%6c%65%72%74%28%64%6f%63%75%6d%65%6e%74%2e%63%6f%6f%6b%69%65%29> [NOVEL]

```

---

## **License**

This project is licensed under the [MIT License](LICENSE). By using, modifying, or distributing this tool, you agree to include the original copyright notice and license.  
For contributions, please submit a pull request, ensuring your changes are in line with the project's goals and adhere to the license terms.

For academic or professional use, please cite this tool as follows:

```plaintext
@misc{xssgai,
  author = {AnonKryptiQuz},
  title  = {XSSGAI: The First Ever AI-Powered XSS Payload Generator},
  year   = {2025},
  url    = {https://github.com/AnonKryptiQuz/XSSGAI}
}
```

---

## **Disclaimer**

- **Ethical Use Only**: XSSGAI is designed for educational, security research, and ethical penetration testing purposes only. The tool should not be used for illegal or malicious activities. Always ensure you have permission to test the website or application.

- **Potential for Improvement**: XSSGAI is an evolving tool, and while it aims to generate valid payloads, it may not always produce perfect results. Users are encouraged to modify parameters to improve performance and generate more accurate payloads. If you make improvements, feel free to contribute by submitting a pull request on GitHub. There are always opportunities for further development and enhancement.

---

## **Credits**

- All tools and libraries are used under their respective open-source licenses.
- The credits for payloads used in this project belong to their rightful owners.

---

## **Author**

**Created by:** [AnonKryptiQuz](https://AnonKryptiQuz.github.io/)
