# Universal_Approximation_Theory

Created using Colab by Saeid Ahmadi.

برنامه زیر ابتدا 100 نقطه نمونه بوسیله یک تابع، اطلاعات ایجاد میکند.

سپس اطلاعات موجود را بوسیله یک تابع آماده سازی میکند.

یک مدل Keras با تابع فعالسازی ReLU و بهینه سازی Adam ایجاد میکند که هرچقدر تعداد گره های آن بیشتر شود به دقت بهتری میرسد،

همچنین تعداد دفعات Epoch تلاش برای برازش بهتر میتواند دقت خروجی مدل را افزایش دهد اما مثلا 10 برابر کردن این عدد مدت زمان اجرای برنامه را 100 برابر میکند در حالی که 10 برابر کردن تعداد گره ها مدت زمان اجرای برنامه را حدودا 2 برابر می کند و به خروجی هایی با همان دقت میرسد.

این مدل با این مقدار اطلاعات ورودی از دقت 0.1 به دقتی حدود 0.01 رسید.

بجای اطلاعات ورودی که بوسیله تابع f(x) ایجاد شد، میتوان از اطلاعات خام استفاده کرد تا این برنامه برای آن اطلاعات یک مدلی بسازد که بتواند بر اساس ورودی های جدید، یک خروجی پیش بینی کند.

نهایتا مدل را برای 3 عدد مختلف امتحان میکنیم که خروجی هایی نزدیک به اعداد خروجی از تابع اولیه خواهد بود.

f(-1.5) =  [-0.95527315]   original f(x)=  -0.625

f(0.5)  =  [-1.1797777]    original f(x)=  -1.125

f(1.8)  =  [6.331234]      original f(x)=  6.272

این برنامه در محیط colab اجرا شد و برای اجرای توابعی از کتابخانه tensorFlow نیاز به CPU است که از AVX پشتیبانی کند یا از GPU استفاده شود، در غیر این صورت دچار خطا خواهد شد و اجرا نمی شود.

The program first generates 100 sample points using a function.

Then it prepares the existing data using a preparation function.

It creates a Keras model with a ReLU activation function and Adam optimization. The more nodes it has, the better its accuracy. The number of Epochs it tries to fit better can also increase the accuracy of the model's output, but for example, increasing this number by 10 times increases the program's execution time by 100 times, while increasing the number of nodes by 10 times doubles the execution time and achieves the same accuracy.

With this amount of input data, the model improved from an accuracy of 0.1 to an accuracy of around 0.01.

Instead of the input data generated by function f(x), raw data can be used so that this program can build a model for that data that can predict an output based on new inputs.

Finally, we try the model for 3 different numbers that will have outputs close to the output numbers from the initial function.

f(x) = x^3 + x^2 - x -1

f(-1.5) = -0.625

f(0.5) = -1.125

f(1.8) = 6.272

This program was run in the Colab environment and requires a CPU that supports AVX or a GPU to run functions from the TensorFlow library. Otherwise, it will error out and not run.
