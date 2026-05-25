require 'json'

package = JSON.parse(File.read(File.join(__dir__, 'package.json')))

Pod::Spec.new do |s|
  s.name = 'VisionchessOffline'
  s.version = package['version']
  s.summary = 'Offline YOLO piece detection for VisionChess (M1)'
  s.license = 'MIT'
  s.homepage = 'https://github.com/sromani/VISIONCHESS'
  s.author = 'VisionChess'
  s.source = { :git => 'https://github.com/sromani/VISIONCHESS.git', :tag => s.version.to_s }
  s.source_files = 'ios/Sources/**/*.{swift,h,m,c,cc,mm,cpp}'
  s.ios.deployment_target = '16.0'
  s.swift_version = '5.9'
  s.dependency 'Capacitor'
  s.dependency 'onnxruntime-objc', '~> 1.19.2'
  s.resource_bundles = { 'VisionchessOfflineResources' => ['ios/Resources/*.onnx'] }
end
