<?php
// send_probe.php - Stupid simple mailer
header('Content-Type: application/json');

// 1. CONFIGURATION
$to = "malte@maltehillebrand.de"; // <--- PUT YOUR EMAIL HERE
$subject = "New Translation Plan Submission";

// 2. GET DATA
$data = json_decode(file_get_contents('php://input'), true);

if (!$data || !isset($data['image'])) {
    echo json_encode(['status' => 'error', 'message' => 'No image data']);
    exit;
}

// 3. DECODE IMAGE
$imageData = $data['image'];
$imageData = str_replace('data:image/png;base64,', '', $imageData);
$imageData = str_replace(' ', '+', $imageData);
$imageContent = base64_decode($imageData);

// 4. PREPARE EMAIL (Multipart)
$boundary = md5(time());
$headers = "From: noreply@t1mesculptures.de\r\n";
$headers .= "MIME-Version: 1.0\r\n";
$headers .= "Content-Type: multipart/mixed; boundary=\"{$boundary}\"\r\n";

// Body
$body = "--{$boundary}\r\n";
$body .= "Content-Type: text/plain; charset=\"UTF-8\"\r\n";
$body .= "Content-Transfer-Encoding: 7bit\r\n\r\n";
$body .= "A new Translation Plan has been submitted. See attachment.\r\n";
$body .= "--{$boundary}\r\n";

// Attachment
$body .= "Content-Type: application/octet-stream; name=\"translation_plan.png\"\r\n";
$body .= "Content-Transfer-Encoding: base64\r\n";
$body .= "Content-Disposition: attachment; filename=\"translation_plan.png\"\r\n\r\n";
$body .= chunk_split(base64_encode($imageContent)) . "\r\n";
$body .= "--{$boundary}--";

// 5. SEND
if (mail($to, $subject, $body, $headers)) {
    echo json_encode(['status' => 'success']);
} else {
    echo json_encode(['status' => 'error', 'message' => 'Mail failed']);
}
?>