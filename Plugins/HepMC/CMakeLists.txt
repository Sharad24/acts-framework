if(USE_HEPMC3)
  file(GLOB src_files "src/*.cpp" "include/ACTFW/Plugins/HepMC/*.hpp")
  add_library(ACTFWPluginHepMC3 SHARED ${src_files})
  target_include_directories(ACTFWPluginHepMC3 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include>)
  target_link_libraries(ACTFWPluginHepMC3 PUBLIC ActsCore ACTFramework)
  target_link_libraries(ACTFWPluginHepMC3 PRIVATE HepMC3Lib) 
  target_link_libraries(ACTFWPluginHepMC3 PRIVATE HepPID)

  install(TARGETS ACTFWPluginHepMC3 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/ACTFW DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
