<?php
$image = $_POST['imageData'];
if (strstr($image,",")){
    $image = explode(',',$image);
    $image = $image[1];
}
$filename = $_POST['filename'];
$r = file_put_contents('images/'.$filename, base64_decode($image));
if (!$r) {

}else{

    $cmd = system(". /home/ubuntu/ML/bin/activate && python transformImage.py ".$filename,$ret);
    $interval=1;//每隔多少秒运行，单位：秒
    $filename = strtr($filename, array('png'=>'json'));
    do{
//这里是你要执行的代码，这里是在一个number.txt的文本里生成数字
        if(file_exists('predicts/'.$filename) == 1){
            break;
        }
//等待执行的时间
        sleep($interval);
    }
    while(true);

    echo file_get_contents('predicts/'.$filename);
}

?>
