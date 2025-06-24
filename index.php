<?php
include 'vendor/autoload.php';

$target_dir = "wrong/";
$fileName = basename($_FILES["captcha"]["name"]);
$fileNameNoExt = substr($fileName,0,-4);
$fileNameExt = substr($fileName,-4,strlen($fileName));
if($fileNameExt != '.png'){
    die();
}
if($fileNameNoExt == ""){
    die();
}

$fileName = 'upload_' . rand(100000,999999) . '.png';

$target_file = $target_dir . $fileName;
$res = move_uploaded_file($_FILES["captcha"]["tmp_name"], $target_file);

$client = new Predis\Client('tcp://10.0.0.11:6379');

$client->publish('captcha',$fileName);

sleep(2);

$text = $client->get($fileName);

rename($target_file, '/var/www/html/predictions/' . $text . '.png');

echo $text;
