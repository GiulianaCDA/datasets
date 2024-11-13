 /* Included by dcerpc_table.c */

#include "dcerpc_spoolss.h"

static const value_string spoolss_opnums[] = {
	{  0,	"OpenPrinter" },
	{  1,	"RemoteFindFirstPrinterChangeNotification" },
	{  2,	"FindClosePrinterChangeNotification" },
	{  3,	"ReadPrinter" },
	{  4,	"WritePrinter" },
	{  5,	"EnumPrinters" },
	{  6,	"AddPrinter" },
	{  7,	"DeletePrinter" },
	{  8,	"SetJob" },
	{  9,	"GetJob" },
	{ 10,	"EnumJobs" },
	{ 11,	"SplOpenPrinter" },
	{ 12,	"AddPrinterDriver" },
	{ 13,	"AddPrinterConnection" },
	{ 14,	"DeletePrinterDriver" },
	{ 15,	"GetPrinterDriverDirectory" },
	{ 16,	"EnumPrinterDrivers" },
	{ 17,	"GetPrinterDriverEx" },
	{ 18,	"AddPrintProcessor" },
	{ 19,	"EnumPrintProcessors" },
	{ 20,	"GetPrintProcessorDirectory" },
	{ 21,	"DeletePrintProcessor" },
	{ 22,	"EnumPrintProcessorDatatypes" },
	{ 23,	"StartDocPrinter" },
	{ 24,	"StartPagePrinter" },
	{ 25,	"WritePrinterEx" },
	{ 26,	"EndPagePrinter" },
	{ 27,	"AbortPrinter" },
	{ 28,	"ReadPrinterEx" },
	{ 29,	"ClosePrinter" },
	{ 30,	"AddJob" },
	{ 31,	"ScheduleJob" },
	{ 32,	"EnumPrintersEx" },
	{ 33,	"GetPrinterDataEx" },
	{ 34,	"SetPrinterDataEx" },
	{ 35,	"DeletePrinterDataEx" },
	{ 36,	"SetPrinterData" },
	{ 37,	"GetPrinterData" },
	{ 38,	"SetJobDelivery" },
	{ 39,	"GetJobDelivery" },
	{ 40,	"AddForm" },
	{ 41,	"DeleteForm" },
	{ 42,	"GetForm" },
	{ 43,	"SetForm" },
	{ 44,	"EnumForms" },
	{ 45,	"EnumMonitors" },
	{ 46,	"AddMonitor" },
	{ 47,	"DeleteMonitor" },
	{ 48,	"ResetPrinter" },
	{ 49,	"GetPrintProcessorCapabilities" },
	{ 50,	"AddPortEx" },
	{ 51,	"StartDocPrinterEx" },
	{ 52,	"AddPort" },
	{ 53,	"ConfigurePort" },
	{ 54,	"DeletePort" },
	{ 55,	"CreatePrinterIC" },
	{ 56,	"PlayGdiScriptOnPrinterIC" },
	{ 57,	"EnumPrinterDataEx" },
	{ 58,	"GetSpoolFileHandle" },
	{ 59,	"CloseSpoolFileHandle" },
	{ 60,	"CommitSpoolData" },
	{ 61,	"RollbackSpoolData" },
	{ 62,	"FlushPrinter" },
	{ 63,	"SendRecvBidiData" },
	{ 64,	"AddPrinterDriverEx" },
	{ 65,	"EnumPrinterKey" },
	{ 66,	"DeletePrinterKey" },
	{ 67,	"SetPortTimeOuts" },
	{ 68,	"XcvOpenPort" },
	{ 69,	"XcvDataPort" },
	{ 70,	"XcvClosePort" },
	{ 71,	"EnumPrinterDriverLanguages" },
	{ 72,	"AddPerMachineConnection" },
	{ 73,	"DeletePerMachineConnection" },
	{ 74,	"EnumPerMachineConnections" },
	{ 75,	"XcvDataPortEx" },
	{ 76,	"AddPrinterConnection2" },
	{ 77,	"SetPrintTickets" },
	{ 78,	"AddMultiPort" },
	{ 79,	"XcvReadRegistryValue" },
	{ 80,	"XcvWriteRegistryValue" },
	{ 81,	"AddPrinterDriverPackage" },
	{ 82,	"DeletePrinterDriverPackage" },
	{ 83,	"GetCorePrinterDrivers" },
	{ 84,	"UninstallPrinterDriverPackage" },
	{ 85,	"EnumCoreDrivers" },
	{ 86,	"AddWsdPrintDriver" },
	{ 87,	"GetDefaultScope" },
	{ 88,	"RegisterEvent" },
	{ 89,	"UnregisterEvent" },
	{ 90,	"RemoteFindFirstPrinterChangeNotificationEx" },
	{ 91,	"EnumPrintDriverLanguages" },
	{ 92,	"GetPrinterDriverPackagePath" },
	{ 93,	"OpenPrinter2" },
	{ 94,	"StartDocPrinterV4" },
	{ 95,	"AddOemUiIcon" },
	{ 96,	"DeleteOemUiIcon" },
	{ 97,	"EnumOemIcons" },
	{ 98,	"QueryOemUiSupport" },
	{ 99,	"GetDocumentAttributes" },
	{100,	"SetDocumentAttributes" },
	{101,	"ResetPrinterEx" },
	{102,	"GetPrinterDriver2" },
	{103,	"AddDriverPackageFromOemInf" },
	{104,	"DownloadGdiScript" },
	{105,	"XcvWebUpdateRegistryValue" },
	{106,	"EnumUmpDevices" },
	{107,	"RegisterForRemoteNotifications" },
	{108,	"GetPrinterDriverPackagePathW" },
	{109,	"QueryRemoteFonts" },
	{110,	"AddPrinterConnection3" },
	{112,	"DeletePortEx" },
	{113,	"EnumDependentServices" },
	{114,	"GetJobNamedPropertyValue" },
	{115,	"SetJobNamedPropertyValue" },
	{116,	"EnumAllJobs" },
	{117,	"DeleteAllJobs" },
	{118,	"QueryAllPrinterDrivers" },
	{119,	"GetCurrentTime" },
	{120,	"SetCurrentTime" },
	{121,	"EnumPortsEx" },
	{122,	"EnumPortDataEx" },
	{123,	"XcvQueryPort" },
	{124,	"XcvEnumeratePorts" },
	{125,	"AddPrinterConnection4" },
	{126,	"SpoolerFindFirstPrinterChangeNotificationEx" },
	{0xffff,	NULL }
};

/* This is the dispatch table for dcerpc_spoolss.c */
static const struct {
	dcerpc_sub_dispatch_fn_t fn;
	const char *name;
} dcerpc_spoolss_table[] = {
#ifdef ENABLE_RPC_SPOOLSS
	{ dcerpc_spoolss_SubDispatch, "SpoolSS" },
#endif
	{ NULL, NULL }
};

static dcerpc_sub_dispatch_fn_t dcerpc_spoolss_SubDispatch(dcerpc_pipe_t *p, uint32_t opnum)
{
	uint16_t nopnum = (uint16_t)(opnum);

	switch (nopnum) {
#ifdef ENABLE_RPC_SPOOLSS
	case DCERPC_spoolss_OpenPrinter:
	case DCERPC_spoolss_SetJob:
	case DCERPC_spoolss_GetJob:
	case DCERPC_spoolss_EnumJobs:
	case DCERPC_spoolss_AddPrinterConnection:
	case DCERPC_spoolss_ClosePrinter:
	case DCERPC_spoolss_StartDocPrinter:
	case DCERPC_spoolss_WritePrinter:
	case DCERPC_spoolss_EndDocPrinter:
	case DCERPC_spoolss_EnumPrinters:
	case DCERPC_spoolss_AddPrinterDriver:
	case DCERPC_spoolss_GetPrinterDriverDirectory:
	case DCERPC_spoolss_DeletePrinterDriverEx:
	case DCERPC_spoolss_AddPrintProcessor:
	case DCERPC_spoolss_EnumPrintProcessors:
	case DCERPC_spoolss_GetPrintProcessorDirectory:
	case DCERPC_spoolss_EnumPrintProcessorDatatypes:
	case DCERPC_spoolss_DeletePrintProcessor:
	case DCERPC_spoolss_StartPagePrinter:
	case DCERPC_spoolss_SendRecvBidiData:
	case DCERPC_spoolss_SetPrinterDataEx:
	case DCERPC_spoolss_GetPrinterDataEx:
	case DCERPC_spoolss_EnumPrinterDataEx:
	case DCERPC_spoolss_DeletePrinterDataEx:
	case DCERPC_spoolss_SetPrinterDriverEx:
	case DCERPC_spoolss_GetPrinterDriverEx:
	case DCERPC_spoolss_EnumPrinterDriversEx:
	case DCERPC_spoolss_DeletePrinterData:
	case DCERPC_spoolss_SetDefaultPrinter:
	case DCERPC_spoolss_GetDefaultPrinter:
	case DCERPC_spoolss_AddForm:
	case DCERPC_spoolss_DeleteForm:
	case DCERPC_spoolss_GetForm:
	case DCERPC_spoolss_SetForm:
	case DCERPC_spoolss_EnumForms:
	case DCERPC_spoolss_EnumMonitors:
	case DCERPC_spoolss_AddMonitor:
	case DCERPC_spoolss_DeleteMonitor:
	case DCERPC_spoolss_ResetPrinter:
	case DCERPC_spoolss_GetPrinterData:
	case DCERPC_spoolss_SetPrinterData:
	case DCERPC_spoolss_AddPrintProvidor:
	case DCERPC_spoolss_DeletePrintProvidor:
	case DCERPC_spoolss_IsValidDevmode:
	case DCERPC_spoolss_GetPrinterDriver:
	case DCERPC_spoolss_GetPrinter:
	case DCERPC_spoolss_SetPrinter:
	case DCERPC_spoolss_AddJob:
	case DCERPC_spoolss_ScheduleJob:
	case DCERPC_spoolss_PrintIsSpoolingActive:
	case DCERPC_spoolss_AcceptNewConnection:
	case DCERPC_spoolss_ReplyOpenPrinter:
	case DCERPC_spoolss_ReplyWritePrinter:
	case DCERPC_spoolss_CloseSpooler:
	case DCERPC_spoolss_FlushPrinter:
	case DCERPC_spoolss_AddFormEx:
	case DCERPC_spoolss_DeleteFormEx:
	case DCERPC_spoolss_GetFormEx:
	case DCERPC_spoolss_SetFormEx:
	case DCERPC_spoolss_EnumFormsEx:
	case DCERPC_spoolss_AddPort:
	case DCERPC_spoolss_ConfigurePort:
	case DCERPC_spoolss_DeletePort:
	case DCERPC_spoolss_GetPrintProcessorCapabilities:
	case DCERPC_spoolss_LoadPrinterDriver:
	case DCERPC_spoolss_AddPerMachineConnection:
	case DCERPC_spoolss_EnumPrintProcessorDatatypesA:
	case DCERPC_spoolss_DeletePrintProvidorEx:
	case DCERPC_spoolss_SetPortTimeOuts:
	case DCERPC_spoolss_XcvOpenPort:
	case DCERPC_spoolss_XcvDataPort:
	case DCERPC_spoolss_XcvClosePort:
	case DCERPC_spoolss_SetSecurityDescriptors:
	case DCERPC_spoolss_GetSecurityDescriptors:
	case DCERPC_spoolss_SpoolerFindFirstPrinterChangeNotification:
	case DCERPC_spoolss_SpoolerFindNextPrinterChangeNotification:
	case DCERPC_spoolss_SpoolerFindClosePrinterChangeNotification:
	case DCERPC_spoolss_CreateLocalPrinterDriver:
	case DCERPC_spoolss_DeleteLocalPrinterDriver:
	case DCERPC_spoolss_AddDirectConnection:
	case DCERPC_spoolss_PrinterMessageBox:
	case DCERPC_spoolss_AddMonitorEx:
	case DCERPC_spoolss_InstallLanguageMonitor:
	case DCERPC_spoolss_AddPortEx:
	case DCERPC_spoolss_StartDocPrinterEx:
	case DCERPC_spoolss_AddPrinterConnection2:
	case DCERPC_spoolss_DeletePrinterDataExA:
	case DCERPC_spoolss_DeletePrinterKeyA:
	case DCERPC_spoolss_SeekPrinter:
	case DCERPC_spoolss_DeleteFormExA:
	case DCERPC_spoolss_GetDefaultDevmode:
	case DCERPC_spoolss_SetPortTimeOutsEx:
	case DCERPC_spoolss_XcvDataPortExA:
	case DCERPC_spoolss_AddPrinterDriverExA:
	case DCERPC_spoolss_InstallPrinterDriverFromPackageA:
	case DCERPC_spoolss_UploadPrinterDriverPackageA:
	case DCERPC_spoolss_GetCorePrinterDriversExA:
	case DCERPC_spoolss_XcvDiagnosticDataPortA:
	case DCERPC_spoolss_CoreReadPrinterA:
	case DCERPC_spoolss_CoreWritePrinterA:
	case DCERPC_spoolss_SetDefaultDevmode:
	case DCERPC_spoolss_DeleteLocalPrinterDriverEx:
	case DCERPC_spoolss_AddDirectConnectionEx:
	case DCERPC_spoolss_GetPrinterDriver2:
	case DCERPC_spoolss_FindFirstPrinterChangeNotificationEx:
	case DCERPC_spoolss_FindNextPrinterChangeNotificationEx:
	case DCERPC_spoolss_AddPrinterDriverExW:
	case DCERPC_spoolss_InstallPrinterDriverFromPackageW:
	case DCERPC_spoolss_UploadPrinterDriverPackageW:
	case DCERPC_spoolss_GetCorePrinterDriversExW:
	case DCERPC_spoolss_XcvDiagnosticDataPortW:
	case DCERPC_spoolss_CoreReadPrinterW:
	case DCERPC_spoolss_CoreWritePrinterW:
	case DCERPC_spoolss_DeleteLocalPrinterDriverExW:
	case DCERPC_spoolss_GetPrinterDriverDirectoryW:
	case DCERPC_spoolss_AddPrintProvidorW:
	case DCERPC_spoolss_DeletePrintProvidorW:
	case DCERPC_spoolss_IsValidDevmodeA:
	case DCERPC_spoolss_GetPrinterDataExA:
	case DCERPC_spoolss_GetPrinterDataWithTypeA:
	case DCERPC_spoolss_EnumPrintersW:
	case DCERPC_spoolss_OpenPrinter2A:
	case DCERPC_spoolss_AddFormA:
	case DCERPC_spoolss_SetFormDataA:
	case DCERPC_spoolss_GetFormA:
	case DCERPC_spoolss_DeleteFormA:
	case DCERPC_spoolss_GetFormFieldsA:
	case DCERPC_spoolss_GetPrinterDriverExA:
	case DCERPC_spoolss_FindClosePrinterChangeNotification:
	case DCERPC_spoolss_ReadRegistryValue:
	case DCERPC_spoolss_WriteRegistryValue:
	case DCERPC_spoolss_DeletePrinterDriverExA:
	case DCERPC_spoolss_AddPrintProcessorA:
	case DCERPC_spoolss_EnumPrintProcessorsA:
	case DCERPC_spoolss_GetPrintProcessorDirectoryA:
	case DCERPC_spoolss_EnumMonitorsA:
	case DCERPC_spoolss_OpenPrinterA:
	case DCERPC_spoolss_ResetPrinterA:
	case DCERPC_spoolss_SetJobA:
	case DCERPC_spoolss_GetJobA:
	case DCERPC_spoolss_EnumJobsA:
	case DCERPC_spoolss_AddPrinterDriverA:
	case DCERPC_spoolss_GetPrinterDriverA:
	case DCERPC_spoolss_GetPrinterA:
	case DCERPC_spoolss_SetPrinterA:
	case DCERPC_spoolss_AddPrinterConnectionA:
	case DCERPC_spoolss_AcceptDarkness:
	case DCERPC_spoolss_RemoteFindFirstPrinterChangeNotification:
	case DCERPC_spoolss_GetPrinterDriverDirectoryW:
	case DCERPC_spoolss_AddPrintProvidorW:
	case DCERPC_spoolss_DeletePrintProvidorW:
	case DCERPC_spoolss_IsValidDevmodeA:
	case DCERPC_spoolss_GetPrinterDataExA:
	case DCERPC_spoolss_GetPrinterDataWithTypeA:
	case DCERPC_spoolss_EnumPrintProcessorDatatypesAsStringA:
	case DCERPC_spoolss_EnumPrintProcessorsW:
	case DCERPC_spoolss_GetPrintProcessorDirectoryW:
	case DCERPC_spoolss_EnumMonitorsW:
	case DCERPC_spoolss_OpenPrinterW:
	case DCERPC_spoolss_ResetPrinterW:
	case DCERPC_spoolss_SetJobW:
	case DCERPC_spoolss_GetJobW:
	case DCERPC_spoolss_EnumJobsW:
	case DCERPC_spoolss_AddPrinterDriverW:
	case DCERPC_spoolss_GetPrinterDriverW:
	case DCERPC_spoolss_GetPrinterW:
	case DCERPC_spoolss_SetPrinterW:
	case DCERPC_spoolss_AddPrinterConnectionW:
	case DCERPC_spoolss_RemoteFindFirstPrinterChangeNotificationEx:
	case DCERPC_spoolss_GetSpoolFileHandle:
	case DCERPC_spoolss_CommitSpoolData:
	case DCERPC_spoolss_CloseSpoolFileHandle:
	case DCERPC_spoolss_FlushPrinter:
	case DCERPC_spoolss_SendRecvBidiData:
	case DCERPC_spoolss_SendRecvBidiDataEx:
	case DCERPC_spoolss_EnumPrinterDataExA:
	case DCERPC_spoolss_EnumPrinterKeyA:
	case DCERPC_spoolss_EnumPrinterDataExW:
	case DCERPC_spoolss_EnumPrinterKeyW:
	case DCERPC_spoolss_DeletePrinterDataExA:
	case DCERPC_spoolss_DeletePrinterDataExW:
	case DCERPC_spoolss_SetPortA:
	case DCERPC_spoolss_SetPortW:
	case DCERPC_spoolss_EnumPrintProcessorDatatypesAsStringW:
	case DCERPC_spoolss_EnumPortsA:
	case DCERPC_spoolss_EnumPortsW:
	case DCERPC_spoolss_AddPrinterDriverExA:
	case DCERPC_spoolss_AddPrinterDriverExW:
	case DCERPC_spoolss_InstallPrinterDriverFromPackageA:
	case DCERPC_spoolss_InstallPrinterDriverFromPackageW:
	case DCERPC_spoolss_UploadPrinterDriverPackageA:
	case DCERPC_spoolss_UploadPrinterDriverPackageW:
	case DCERPC_spoolss_GetCorePrinterDriversA:
	case DCERPC_spoolss_GetCorePrinterDriversW:
	case DCERPC_spoolss_CorePrinterDriverInstalledA:
	case DCERPC_spoolss_CorePrinterDriverInstalledW:
	case DCERPC_spoolss_GetPrinterDriverPackagePathA:
	case DCERPC_spoolss_GetPrinterDriverPackagePathW:
	case DCERPC_spoolss_DeletePrinterDriverPackageA:
	case DCERPC_spoolss_DeletePrinterDriverPackageW:
	case DCERPC_spoolss_AddPerMachineConnectionA:
	case DCERPC_spoolss_AddPerMachineConnectionW:
	case DCERPC_spoolss_DeletePerMachineConnectionA:
	case DCERPC_spoolss_DeletePerMachineConnectionW:
	case DCERPC_spoolss_EnumPerMachineConnectionsA:
	case DCERPC_spoolss_EnumPerMachineConnectionsW:
	case DCERPC_spoolss_XcvDataW:
	case DCERPC_spoolss_AddPrinterDriverPackageA:
	case DCERPC_spoolss_AddPrinterDriverPackageW:
	case DCERPC_spoolss_GetPrintSchemaVersion:
	case DCERPC_spoolss_FindFirstPrinterChangeNotificationEx:
	case DCERPC_spoolss_Stress:
	case DCERPC_spoolss_RegisterForRemoteWmiEvents:
	case DCERPC_spoolss_FindClosePrinterChangeNotificationEx:
	case DCERPC_spoolss_AddPortExA:
	case DCERPC_spoolss_AddPortExW:
	case DCERPC_spoolss_StartDocPrinterW:
	case DCERPC_spoolss_StartPagePrinter:
	case DCERPC_spoolss_WritePrinter:
	case DCERPC_spoolss_EndPagePrinter:
	case DCERPC_spoolss_AbortPrinter:
	case DCERPC_spoolss_ReadPrinter:
	case DCERPC_spoolss_EndDocPrinter:
	case DCERPC_spoolss_AddJobW:
	case DCERPC_spoolss_ScheduleJob:
	case DCERPC_spoolss_AddFormW:
	case DCERPC_spoolss_SetFormDataW:
	case DCERPC_spoolss_GetFormW:
	case DCERPC_spoolss_DeleteFormW:
	case DCERPC_spoolss_GetFormFieldsW:
	case DCERPC_spoolss_XcvDataA:
	case DCERPC_spoolss_AddPrinterDriverExW:
	case DCERPC_spoolss_InstallPrinterDriverFromPackageW:
	case DCERPC_spoolss_UploadPrinterDriverPackageW:
	case DCERPC_spoolss_GetCorePrinterDriversW:
	case DCERPC_spoolss_CorePrinterDriverInstalledW:
	case DCERPC_spoolss_GetPrinterDriverPackagePathW:
	case DCERPC_spoolss_DeletePrinterDriverPackageW:
	case DCERPC_spoolss_AddPerMachineConnectionW:
	case DCERPC_spoolss_DeletePerMachineConnectionW:
	case DCERPC_spoolss_EnumPerMachineConnectionsW:
	case DCERPC_spoolss_AddPrinterDriverPackageW:
	case DCERPC_spoolss_FindFirstPrinterChangeNotificationExA:
	case DCERPC_spoolss_FindFirstPrinterChangeNotificationExW:
	case DCERPC_spoolss_AddPortExW:
	case DCERPC_spoolss_StartDocPrinterA:
	case DCERPC_spoolss_AddJobA:
	case DCERPC_spoolss_InstallPrinterDriverFromPackageA:
	case DCERPC_spoolss_UploadPrinterDriverPackageA:
	case DCERPC_spoolss_GetCorePrinterDriversA:
	case DCERPC_spoolss_CorePrinterDriverInstalledA:
	case DCERPC_spoolss_GetPrinterDriverPackagePathA:
	case DCERPC_spoolss_DeletePrinterDriverPackageA:
	case DCERPC_spoolss_AddPerMachineConnectionA:
	case DCERPC_spoolss_DeletePerMachineConnectionA:
	case DCERPC_spoolss_EnumPerMachineConnectionsA:
	case DCERPC_spoolss_AddPrinterDriverPackageA:
	case DCERPC_spoolss_AddFormA:
	case DCERPC_spoolss_SetFormDataA:
	case DCERPC_spoolss_GetFormA:
	case DCERPC_spoolss_DeleteFormA:
	case DCERPC_spoolss_GetFormFieldsA:
	case DCERPC_spoolss_AddPrinterDriverExA:
	case DCERPC_spoolss_AddPortExA:
	case DCERPC_spoolss_StartDocPrinterA,
	case DEVICE_CAPS:
	case GETSETPAPERBINS:
	case DEVICERESOURCE:
	case DOCUMENTPROPERTIESA:
	case PRINTERTASKSETTINGSA:
	case GETTECHNOLOGY:
	case NEXTBANDINFO:
	case SIDETOSIDE:
	case BIDIRECTIONALSCHEMA:
	case GETSETPAPERMETRICSA:
	case GETDEVICEUNITS:
	case QUERYDIBSUPPORT:
	case BEGINPATH:
	case CLIPTOPATH:
	case ENDPATH:
	case EXTFLOODFILL:
	case FLOODFILL:
	case RECTVISIBLE:
	case DRAWPATTERNRECT:
	case ENUMPOLYGONREGION:
	case FILLREGION:
	case FRAMERGN:
	case GETTEXTCHARACTEREXTRA:
	case GETASPECTRATIOFILTEREX:
	case INVERTRGN:
	case PAINTRGN:
	case PATHTOREGION:
	case POLYGONREGION:
	case PTINREGION:
	case ENUMOBJECTS:
	case EXTTEXTOUTA:
	case GETTEXTEXTENTPOINTA:
	case GETEXTENDEDTEXTMETRICSA:
	case GETFACENAMEA:
	case GETTEXTCHARACTEREXTRA,
	case GETSETPAPERBINS,
	case DEVICERESOURCE,
	case DOCUMENTPROPERTIESA,
	case PRINTERTASKSETTINGSA,
	case GETTECHNOLOGY,
	case NEXTBANDINFO,
	case SIDETOSIDE,
	case BIDIRECTIONALSCHEMA,
	case GETSETPAPERMETRICSA,
	case GETDEVICEUNITS,
	case QUERYDIBSUPPORT,
	case BEGINPATH,
	case CLIPTOPATH,
	case ENDPATH,
	case EXTFLOODFILL,
	case FLOODFILL,
	case RECTVISIBLE,
	case DRAWPATTERNRECT,
	case ENUMPOLYGONREGION,
	case FILLREGION,
	case FRAMERGN,
	case GETTEXTCHARACTEREXTRA,
	case GETASPECTRATIOFILTEREX,
	case INVERTRGN,
	case PAINTRGN,
	case PATHTOREGION,
	case POLYGONREGION,
	case PTINREGION,
	case ENUMOBJECTS,
	case EXTTEXTOUTA,
	case GETTEXTEXTENTPOINTA,
	case GETEXTENDEDTEXTMETRICSA,
	case GETFACENAMEA,
	case DOCUMENTPROPERTIESW:
	case PRINTERTASKSETTINGSW:
	case DEVICEDATA:
	case DRVQUERYDEVICESUPPORT:
	case GETSETPRINTORIENT:
	case ENUMPOLYGONREGION,
	case FILLREGION,
	case FRAMERGN,
	case INVERTRGN,
	case PAINTRGN,
	case PATHTOREGION,
	case POLYGONREGION,
	case PTINREGION,
	case ENUMOBJECTS,
	case EXTTEXTOUTW:
	case GETTEXTEXTENTPOINTW:
	case GETEXTENDEDTEXTMETRICSW:
	case GETFACENAMEW:
	case SETPAPERMETRICS,
	case SETDIBITS:
	case SETDIBITSTODEVICE:
	case STRETCHDIB:
	case STRETCHBLT:
	case SETSYSTEM paletteuse,
	case SETCOLORADJUSTMENT,
	case SETCOLORSPACE,
	case DELETEOBJECTAPP,
	case CREATEPALETTE,
	case ANIMATEPALETTE,
	case RESIZEPALETTE,
	case REALIZEOBJECT,
	case UPDATECOLORSINMETAFILE:
	case ENABLERELATIVEWIDTHS,
	case SETMAGICCOLORS,
	case SETLAYOUTWIDTH,
	case GETDEVICEGDIPHANDLE:
	case GETNEARESTPALETTEINDEX:
	case GETLOGCOLORSPACEA:
	case GETLOGCOLORSPACEW:
	case COUNTSOLIDCOLORS:
	case GDI_DRAWINGGROUP:
	case ADDTHOUSANDSEPARATOR,
	case SETCALENDARINFOA:
	case QUERYCALENDARINFOA:
	case SETCALENDARINFOW:
	case QUERYCALENDARINFOW:
	case MULDIVROUND:
	case GETSYSTEMTIMEADJUSTMENTP,
	case SETSYSTEMTIMEADJUSTMENTP:
	case ISVALIDLANGUAGENAMEA,
	case ISVALIDLANGUAGENAMEW,
	case ISVALIDLOCALEINFO,
	case ISVALIDCODEPAGE,
	case ISNORMALIZEDSTRING,
	case GETFILEVERSIONINFOSIZEA,
	case GETFILEVERSIONINFOSIZEW:
	case GETFILEVERSIONINFOA,
	case GETFILEVERSIONINFOW,
	case VERQUERYVALUEA,
	case VERQUERYVALUEW,
	case FINDRESOURCEEXA,
	case FINDRESOURCEEXW,
	case ENUMRESLANGUAGESA,
	case ENUMRESLANGUAGESW,
	case LOADSTRINGA,
	case LOADSTRINGW:
		// Windows API functions not directly related to printing can be handled here.
		// For example, you might want to define behavior for specific system calls or resource handling.
		break;

	default:
		std::cout << "Unknown message\n";
		break;
	}
}