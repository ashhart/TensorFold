Pod::Spec.new do |s|
  s.name = 'TensorFoldWatchConnectivity'
  s.version = '1.0.0'
  s.summary = 'Apple Watch status and reply relay'
  s.description = s.summary
  s.license = { :type => 'UNLICENSED' }
  s.author = 'TensorFold'
  s.homepage = 'https://example.com'
  s.source = { :git => 'https://example.com/tensorfold-mobile.git' }
  s.platforms = { :ios => '15.1' }
  s.swift_version = '5.9'
  s.static_framework = true
  s.dependency 'ExpoModulesCore'
  s.dependency 'TensorFoldPinnedNetwork'
  s.frameworks = 'WatchConnectivity', 'Security', 'UIKit'
  s.source_files = '**/*.swift'
end
