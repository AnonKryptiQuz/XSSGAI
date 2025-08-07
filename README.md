# **XSSGAI: The First Ever AI-Powered XSS Payload Generator**

**XSSGAI** is the **first-ever AI-powered XSS (Cross-Site Scripting) payload generator**. Leveraging machine learning and deep learning techniques, it creates novel payloads based on patterns learned from a dataset of real-world XSS attacks. This groundbreaking tool is designed to assist security researchers and ethical hackers in identifying and mitigating potential XSS vulnerabilities by generating diverse and sophisticated attack vectors.

---

## **Features**

- **AI-Generated Payloads**: Uses advanced neural networks trained on a dataset of real-world XSS attacks to generate innovative payloads. The model achieves a **Validation Accuracy of 91.42%** and a **Validation F1 Score of 72.96%**, indicating reliable payload generation with balanced performance across both precision and recall.
  
- **Large and Comprehensive Dataset**: Trained on a dataset containing **14,437 training payloads** and **3,609 testing payloads**, providing a robust foundation for generating diverse and sophisticated XSS attack vectors.

- **Temperature Sampling**: Allows generation of payloads with varying levels of creativity and randomness, enabling users to fine-tune the output based on their needs.

- **SentencePiece Tokenization**: Employs efficient tokenization for payload generation, ensuring flexibility and adaptability while handling complex patterns in the dataset.

- **Customizable Parameters**: Adjust parameters like temperature, max length, and repetition limits to fine-tune payload generation according to specific requirements.

- **Google Colab Support**: Easily run the tool in Google Colab for cloud-based execution without local setup, making it accessible to users without high-end hardware.

- **Open Source**: Fully open-source under the MIT License, encouraging community contributions and improvements.

---

## **Prerequisites**

To use this tool, ensure you have the following installed:

- **Python 3.x**
- **TensorFlow**
- **SentencePiece**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib** and **Seaborn** (for analysis and visualization)

If you’re using Google Colab, all dependencies will be installed automatically during setup.

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
   tensorflow
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   sentencepiece
   wordcloud
   requests
   keras
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
    * *Optional:* You can also upload the `best_model.keras` file to save time and skip the training process if you'd like to continue testing from a pre-trained model.

3. **Run the Notebook**:  
   - Execute all the cells sequentially by selecting **Runtime > Run All** or by running each cell individually.
   - The notebook will automatically handle dataset processing, install necessary dependencies, train the model, and generate the payloads.

4. **View Results**:  
   - After running the cells, the generated payloads and any associated outputs (e.g., graphs, metrics) will be displayed directly in the notebook.
   - You can save the results or export them as needed.

---

## **Usage**

1. **Set Parameters**:  
   Adjust the following parameters in the notebook to customize payload generation and optimize model performance:
   
   - **max_sequence_length**: Set to 3098 (default value)
   - **vocab_size**: Set to 175 (default value)
   - **new_max_sequence_length**: Set to 100
   - **learning_rate**: Set to 0.001
   - **epochs**: Set to 10
   - **patience**: Set to 5
   - **Dropout**: Set to 0.2
   - **batch_size**: Set to 64
   - **seq_length**: Set to 70
   - **GRU units**: Set to 64
   - **output_dim**: Set to 64
   - **train_inputs**: Set to 200,000
   - **max_length**: Set to 70
   - **max_repeats**: Set to 3
   - **temperature**: Set to 0.3
   - **model_name**: Set to "best_model.keras"
   
   Modify these values as needed to experiment with different settings and observe how they impact the accuracy and performance of the model.

2. **Modify Payload Seed Text**:  
   You can change the initial payload by adjusting the seed text. For example, you can set:
   - **seed_text**: `"<script>alert(\"AnonKryptiQuz\")"`
   
   This allows you to explore how variations in the seed text influence the generated payloads.

3. **Experiment & Evaluate**:  
   Feel free to adjust these or other parameters and observe how changes impact model accuracy, payload generation, and overall performance. This flexibility allows you to fine-tune the generator for optimal results.

4. **Viewing Results Without Downloading Files**:  
   If you prefer to view the results without downloading any files, you can simply view the generated **PDF file**: [XSSGAI_AnonKryptiQuz.pdf](./XSSGAI_AnonKryptiQuz.pdf), which contains all relevant output.

---

## **Example Output**

```plaintext
Generated Payload (Temperature: 0.1): <script>alert("AnonKryptiQuz");</script>
Generated Payload (Temperature: 0.3): <script>alert("AnonKryptiQuz")</script>
Generated Payload (Temperature: 0.5): <script>alert("AnonKryptiQuz");</script>
Generated Payload (Temperature: 1.0): <script>alert("AnonKryptiQuz")</script>
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
