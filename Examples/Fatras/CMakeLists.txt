set(_common_libraries
  ActsCore
  FatrasCore
  ACTFramework
  ACTFWBFieldPlugin
  ACTFWCsvPlugin
  ACTFWDigitization
  ACTFWExamplesCommon
  ACTFWFatras
  ACTFWObjPlugin
  ACTFWRootPlugin
  ActsFrameworkGenerators
  ActsFrameworkPythia8)

# Generic detector
add_executable(ACTFWGenericFatrasExample src/GenericFatrasExample.cpp)
target_include_directories(ACTFWGenericFatrasExample PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(ACTFWGenericFatrasExample PRIVATE ActsCore FatrasCore)
target_link_libraries(ACTFWGenericFatrasExample PRIVATE ACTFramework ACTFWExamplesCommon ActsFrameworkGenerators)
target_link_libraries(ACTFWGenericFatrasExample PRIVATE ACTFWFatras ACTFWDigitization)
target_link_libraries(ACTFWGenericFatrasExample PRIVATE ACTFWBFieldPlugin ActsFrameworkPythia8)
target_link_libraries(ACTFWGenericFatrasExample PRIVATE ACTFWObjPlugin ACTFWRootPlugin ACTFWCsvPlugin)
target_link_libraries(ACTFWGenericFatrasExample PRIVATE ACTFWGenericDetector)
target_link_libraries(ACTFWGenericFatrasExample PRIVATE ${Boost_LIBRARIES})
install(TARGETS ACTFWGenericFatrasExample RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# TGEO based detector
if (USE_TGEO)
  add_executable(ACTFWRootFatrasExample src/RootFatrasExample.cpp)
  target_include_directories(ACTFWRootFatrasExample PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(ACTFWRootFatrasExample PRIVATE ActsCore FatrasCore)
  target_link_libraries(ACTFWRootFatrasExample PRIVATE ACTFramework ACTFWExamplesCommon ActsFrameworkGenerators)
  target_link_libraries(ACTFWRootFatrasExample PRIVATE ACTFWFatras ACTFWDigitization)
  target_link_libraries(ACTFWRootFatrasExample PRIVATE ACTFWBFieldPlugin ActsFrameworkPythia8)
  target_link_libraries(ACTFWRootFatrasExample PRIVATE ACTFWObjPlugin ACTFWRootPlugin ACTFWCsvPlugin)
  target_link_libraries(ACTFWRootFatrasExample PRIVATE ACTFWRootDetector)
  target_link_libraries(ACTFWRootFatrasExample PRIVATE ${Boost_LIBRARIES})
  install(TARGETS ACTFWRootFatrasExample RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# DD4hep detector
if (USE_DD4HEP)
  add_executable(ACTFWDD4hepFatrasExample src/DD4hepFatrasExample.cpp)
  target_include_directories(ACTFWDD4hepFatrasExample PRIVATE ${DD4hep_INCLUDE_DIRS})
  target_include_directories(ACTFWDD4hepFatrasExample PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(ACTFWDD4hepFatrasExample PRIVATE ActsCore FatrasCore)
  target_link_libraries(ACTFWDD4hepFatrasExample PRIVATE ACTFramework ACTFWExamplesCommon ActsFrameworkGenerators)
  target_link_libraries(ACTFWDD4hepFatrasExample PRIVATE ACTFWFatras ACTFWDigitization)
  target_link_libraries(ACTFWDD4hepFatrasExample PRIVATE ACTFWBFieldPlugin ActsFrameworkPythia8)
  target_link_libraries(ACTFWDD4hepFatrasExample PRIVATE ACTFWObjPlugin ACTFWRootPlugin ACTFWCsvPlugin)
  target_link_libraries(ACTFWDD4hepFatrasExample PRIVATE ACTFWDD4hepDetector)
  target_link_libraries(ACTFWDD4hepFatrasExample PRIVATE ${DD4hep_LIBRARIES})
  target_link_libraries(ACTFWDD4hepFatrasExample PRIVATE ${Boost_LIBRARIES})
  install(TARGETS ACTFWDD4hepFatrasExample RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
