Download Link: https://assignmentchef.com/product/solved-cs771-homework-1
<br>
Machine Learning )

<strong>Instructions:</strong>

<ul>

 <li>Only electronic submissions will be accepted. Your main PDF writeup must be typeset in LaTeX (please also refer to the “Additional Instructions” below).</li>

 <li>The PDF writeup containing your solution has to be submitted via Gradescope <a href="https://www.gradescope.com/">https://www.gradescope. </a><u>com/</u><u>.</u></li>

 <li>We have created your Gradescope account (you should have received the notification). Please use yourIITK CC ID (not any other email ID) to login. Use the “Forgot Password” option to set your password.</li>

</ul>

<strong>Additional Instructions</strong>

<ul>

 <li>We have provided a LaTeX template file tex to help typeset your PDF writeup. There is also a style file ml.sty that contain shortcuts to many of the useful LaTeX commends for doing things such as boldfaced/calligraphic fonts for letters, various mathematical/greek symbols, etc., and others. Use of these shortcuts is recommended (but not necessary).</li>

 <li>Your answer to every question should begin on a new page. The provided template is designed to do this However, if it fails to do so, use the clearpage option in LaTeX before starting the answer to a new question, to <em>enforce </em>this.</li>

 <li>While submitting your assignment on the Gradescope website, you will have to specify on which page(s) is question 1 answered, on which page(s) is question 2 answered etc. To do this properly, first ensure that the answer to each question starts on a different page.</li>

 <li>Be careful to flush all your floats (figures, tables) corresponding to question n before starting the answer to question n + 1 otherwise, while grading, we might miss your important parts of your answers.</li>

 <li>Your solutions must appear in proper order in the PDF file i.e. solution to question n must be complete in the PDF file (including all plots, tables, proofs etc) before you present a solution to question n + 1.</li>

</ul>




<strong>Problem 1 (15 marks)</strong>

<strong>(Absolute Loss Regression with Sparsity)</strong> The absolute loss regression problem with i1 regularization is




<table>

 <tbody>

  <tr>

   <td width="621">

    <table width="100%">

     <tbody>

      <tr>

       <td>

        <table>

         <tbody>

          <tr>

           <td width="287">wopt = argmin<em>W</em></td>

           <td width="24">~Nn=1</td>

           <td width="306">|y<sub>n</sub> − w<sup>T</sup>x<sub>n</sub>| + A||w||1</td>

          </tr>

         </tbody>

        </table> </td>

      </tr>

     </tbody>

    </table></td>

  </tr>

 </tbody>

</table>




where ||w||1 = &gt;Dd=1 |wd|, |.| is the absolute value function, and A &gt; 0 is the regularization hyperparameter.

Is the above objective function convex? You don’t need to prove this formally; just a brief reasoning based on properties of other functions that are known to be convex/non-convex would be fine.

Derivate the expression for the (sub)gradient vector for this model.

<strong>Problem 2 (15 marks)</strong>

<strong>(Feature Masking as Regularization)</strong> Consider linear regression model by minimizing the squared loss func­tion &gt;Nn=1(yn −<sub> w</sub>T<sub>xn</sub>)<sup>2</sup>. Suppose we decide to mask out or “drop” each feature Xnd of each input x<sub>n</sub> ∈ R<sup>D</sup>, independently, with probability 1 − p (equivalently, retaining the feature with probability p). Masking or drop­ping out basically means that we will set the feature Xnd to 0 with probability 1 − p. Essentially, it would be equivalent to replacing each input x<sub>n</sub> by <sup>˜</sup>x<sub>n</sub> = x<sub>n</sub> ◦ m<sub>n</sub>, where ◦ denotes elementwise product and m<sub>n</sub> denotes the D ×1 binary mask vector with mnd ∼ Bernoulli(p) (mnd = 1 means the feature Xnd was retained; mnd = 0 means the feature Xnd was masked/zeroed).

Let us now define a new loss function using these masked inputs as follows: &gt;N n=1(yn − w<sup>T˜</sup>x<sub>n</sub>)<sup>2</sup>. Show that minimizing the<em> expected</em> value of this new loss function (where the expectation is used since the mask vectors m<sub>n</sub> are random) is equivalent to minimizing a<strong> regularized</strong> loss function. Clearly write down the expression of this regularized loss function.

<strong>Problem 3 (40 marks)</strong>

<strong>(Multi-output Regression with Reduced Number of Parameters)</strong> Consider the multi-output regression in which each output y<sub>n</sub> ∈ RM in a real-valued vector, rather than a scalar. Assuming a linear model, we can model the outputs as Y ≈ XW, where X is the N × D feature matrix and Y is N × M response<em> matrix</em> with row n being y<sup>T</sup><sub>n</sub> (note that each column of Y denotes one of the M responses), and W is the D × M<strong> weight</strong> <strong>matrix</strong>, with its M columns containing the M weight vectors w1, w2,. . . , wM. Let’s define a squared error loss function &gt;j=1&gt;=1(ynm−wxn)2, which is just the usual squared error but summed over all the M outputs. Firstly, verify that this can also be written in a more compact notation as TRACE[(Y − XW)<sup>T</sup>(Y − XW)].

Further, we will assume that the weight matrix W can be written as a product of two matrices, i.e., W = BS where B is D × K and S is K × M (assume K &lt; min{D, M}). Note that there is a benefit of modeling W this way, since now we need to learn only K × (D + M) parameters as opposed to D × M parameters and, if K is small, this can significantly reduce the number of parameters (in fact, reducing the<em> effective</em> number of parameters to be learned is another way of regularizing a machine learning model). Note (you can verify) that in this formulation, each wm can be written as a linear combination of K columns of B.

With the proposed representation of W, the new objective will be TRACE[(Y − XBS)T(Y − XBS)] and you need to learn both B and S by solving the following problem:




<table>

 <tbody>

  <tr>

   <td width="621">

    <table width="100%">

     <tbody>

      <tr>

       <td>

        <table>

         <tbody>

          <tr>

           <td width="169">{</td>

           <td width="103"><sup>ˆ</sup>B, <sup>ˆ</sup>S}= arg minB,S</td>

           <td width="345">TRACE[(Y − XBS)T(Y − XBS)]</td>

          </tr>

         </tbody>

        </table> </td>

      </tr>

     </tbody>

    </table></td>

  </tr>

 </tbody>

</table>




We will ignore regularization on B and S for brevity/simplicity.




Derive an alternating optimization (ALT-OPT) algorithm to learn B and S, clearly writing down the expressions for the updates of B and S. Are both subproblems (solving for B and solving for S) equally easy/difficult in this ALT-OPT algorithm? If yes, why? If no, why not?

Note: Since B and S are matrices, if you want, please feel free to use results for matrix derivatives (results you will need can be found in Sec. 2.5 of the Matrix Cookbook). However, the problem can be solved even without using matrix derivative results with some rearragement of terms and using vector derivatives.

<strong>Problem 4 (10 marks)</strong>

<strong>Ridge Regression using Newton’s Method </strong>Consider the ridge regression problem:




<table>

 <tbody>

  <tr>

   <td width="652">

    <table width="100%">

     <tbody>

      <tr>

       <td>

        <table>

         <tbody>

          <tr>

           <td width="131">wˆ = arg min<strong>w</strong></td>

           <td width="11">12</td>

           <td width="24">~Nn=1</td>

           <td width="482">(y<sub>n</sub> – w<sup>T</sup>x<sub>n</sub>)<sup>2</sup> + λ 2 w<sup>T</sup>w = arg min<sub>2</sub>(y – Xw)<sup>T</sup>(y – Xw) + λ1                                                                                                2 wTw<strong>w</strong></td>

          </tr>

         </tbody>

        </table> </td>

      </tr>

     </tbody>

    </table></td>

  </tr>

 </tbody>

</table>




where X is the N x D feature matrix and y is the N x 1 vector of labels of the N training examples. Note that the factor of <sub>2</sub><sup> 1</sup>has been used in the above expression just for convenience of derivations required for this problem and does not change the solution to the problem.

Derive the Newton’s method’s update equations for each iteration. For this model, how many iterations would the Newton’s method will take to converge?

<strong>Problem 5 (20 marks)</strong>

<strong>(Dice Roll) </strong>You have a six-faced dice which you roll N times and record the number of times each of its six faces are observed. Suppose these numbers are N1, N2,. . . , N6, respectively. Assume that the probability of a random roll of the dice showning the k<sup>th</sup> face (k = 1,2,. . . , 6) to be equal to 7rk E (0, 1).

Assuming an appropriate conjugate prior for the probability vector π = [7r1, 7r2, . . . , 7r6], derive its MAP esti­mate. In which situation(s), you would expect the MAP solution to be better than the MLE solution?

Also derive the full posterior distribution over π using the same prior that you used for MAP estimate. Given this posterior, can you get the MLE and MAP estimate without solving the MLE and MAP optimization problems? If yes, how? If no, why not?